import codecs
from collections import defaultdict, namedtuple
import torch
from tqdm import tqdm
from utils.data_reader import PAD_ID, BOS_ID, EOS_ID


class ARPAConverter:
    def __init__(self, word2idx, model, rnn_ngram_context=3, top_ngrams={2: 2000000, 3: 300000}):
        self._model = model

        self._word2idx = word2idx
        self._idx2word = {v:k for k,v in self._word2idx.items()}

        self._base_unigrams = {}
        self._base_bigrams = {}

        self._avg_unigram_backoff = 0
        self._avg_bigram_backoff = 0

        self._rnn_ngram_context = rnn_ngram_context
        self._rnn_ngrams = {}
        for n in range(2, rnn_ngram_context+1):
            self._rnn_ngrams[n] = defaultdict(lambda: [0, 0])

        self._rnn_history = {}
        for n in range(2, rnn_ngram_context+1):
            self._rnn_ngrams[n] = defaultdict(set)

    def read_lm(self, lm_file):
        marker = 'start'

        unigram_cnt, bigram_cnt = 0, 0
        dict_val = namedtuple('DictVal', ['log_prob', 'backoff_weight'])
        with codecs.open(lm_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if line:
                    if marker == 'start':
                        if line == '\\1-grams:':
                            marker = 'uni'
                    elif marker == 'uni':
                        if line != '\\2-grams:':
                            try:  # catch invalid backoffs (</s>, <unk>)
                                log_prob, unigram, backoff = line.split()
                                try:
                                    self._base_unigrams[unigram] = dict_val(float(log_prob), float(backoff))
                                    self._avg_unigram_backoff += float(backoff)
                                    unigram_cnt += 1
                                except KeyError:
                                    pass
                            except ValueError:
                                pass
                        else:
                            marker = 'bi'
                    elif marker == 'bi':
                        if line != '\\3-grams:':
                            try:  # more catching
                                prob, history, word, backoff = line.split()
                                try:
                                    bigram = (history, word)
                                    self._base_bigrams[bigram] = dict_val(float(log_prob), float(backoff))
                                    self._avg_bigram_backoff += float(backoff)
                                    bigram_cnt += 1
                                except KeyError:
                                    pass
                            except ValueError:
                                pass
                        else:
                            marker = "tri"
                    elif marker == "tri":
                        continue

        self._avg_unigram_backoff /= unigram_cnt
        self._avg_bigram_backoff /= bigram_cnt

    def _update_rnn_ngram_prob_batch(self, batch_data, batch_output):
        for n in range(2, self._rnn_ngram_context + 1):
            for sent_id, sent in enumerate(batch_data):
                for curr_word_pos, curr_word_idx in enumerate(sent):
                    if curr_word_idx == PAD_ID or curr_word_idx == EOS_ID:
                        break
                    else:
                        history = batch_data[sent_id][curr_word_pos - n + 2:curr_word_pos + 1]
                        history = history.cpu().tolist()
                        if not history:
                            continue
                        history = [self._idx2word[word_idx] for word_idx in history]

                        output_probs = batch_output[sent_id][curr_word_pos]
                        for pred_word_idx, pred_word_log_prob in enumerate(output_probs):
                            if pred_word_idx == PAD_ID or pred_word_idx == BOS_ID:
                                continue
                            pred_word = self._idx2word[pred_word_idx]
                            self._rnn_ngram_counts[n][tuple(history + [pred_word])][0] += pred_word_log_prob.item()
                            self._rnn_ngram_counts[n][tuple(history + [pred_word])][1] += 1
                            self._rnn_history[tuple(history)].add(pred_word)

    def _get_label_probabilities(self, data_iter):
        self._model.eval()
        with torch.no_grad():
            for batch in tqdm(data_iter):
                batch_data, batch_targets, batch_pad_lengths = batch[0], batch[1], batch[2]
                batch_output = self._model(batch_data, batch_pad_lengths)

                self._update_rnn_ngram_prob_batch(batch_data, batch_output)
                del batch_data
                del batch_targets
                del batch_pad_lengths
                del batch_output

        # Average and Normalization
        for n in self._rnn_ngrams:
            prob_sum = 0
            for n_gram in self._rnn_ngram_counts[n]:
                self._rnn_ngram_counts[n][n_gram][0] /= self._rnn_ngrams[n][n_gram][1]
                prob_sum += self._rnn_ngram_counts[n][n_gram][0]
            for n_gram in self._rnn_ngram_counts[n]:
                self._rnn_ngrams[n][n_gram][0] /= prob_sum

    def write_arpa_format(self, file_path):
        model_file = codecs.open(file_path, "w+", encoding="utf8")

        print("\\data\\", file=model_file)
        print("ngram 1={0}".format(len(self._base_unigrams)), file=model_file)
        print("ngram 2={0}".format(len(self._rnn_ngrams[2])), file=model_file)
        print("ngram 3={0}".format(len(self._rnn_ngrams[3])), file=model_file)

        print("\n\\1-grams:", file=model_file)
        for unigram in self._base_unigrams:
            if unigram == '<s>' or unigram == '<\s>':
                print("{0} {1}".format(self._base_unigrams[unigram].log_prob,
                                       unigram),
                      file=model_file)
            else:
                try:  # try/catch for log(0)
                    print("{0} {1} {2}".format(self._base_unigrams[unigram].log_prob,
                                               unigram,
                                               self._base_unigrams[unigram].backoff_weight, file=model_file))
                except ValueError:
                    print("{0} {1} {2}".format(-99,
                                               unigram,
                                               self._base_unigrams[unigram].backoff_weight),
                          file=model_file)  # ARPA standard substitution for log(0) is -99 according to documentation

        print("\n\\2-grams:", file=model_file)
        for bigram in self._rnn_ngrams[2]:
            try:
                print("{0} {1} {2} {3}".format(self._rnn_ngrams[2][bigram],
                                               bigram[0],
                                               bigram[1],
                                               self._base_bigrams[bigram]),
                      file=model_file)
            except ValueError:
                print("{0} {1} {2}".format(-99,
                                           bigram[0],
                                           bigram[1],
                                           self._base_bigrams[bigram].backoff_weight),
                      file=model_file)
            except KeyError:
                print("{0} {1} {2}".format(self._rnn_ngrams[2][bigram],
                                           bigram[0],
                                           bigram[1],
                                           self._avg_bigram_backoff),
                      file=model_file)

        print("\n\\3-grams:", file=model_file)
        for trigram in self._rnn_ngrams[3]:
            try:
                print("{0} {1} {2} {3}".format(self._rnn_ngrams[3][trigram][0],
                                               trigram[0],
                                               trigram[1],
                                               trigram[2]),
                      file=model_file)
            except ValueError:
                print("{0} {1} {2} {3}".format(-99,
                                               trigram[0],
                                               trigram[1],
                                               trigram[2]),
                      file=model_file)

        print("\n\\end\\", file=model_file)
        model_file.close()
