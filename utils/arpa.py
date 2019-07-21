import codecs
from collections import defaultdict, namedtuple
import logging
import torch
from tqdm import tqdm
from utils.data_reader import PAD_ID, BOS_ID, EOS_ID
from utils.trainer import sort_by_lengths


class ARPAConverter:
    def __init__(self, idx2word, model, rnn_ngram_context=3,
                 ngram_pruning={2: 2000000, 3: 300000}, history_pruning={2:50000}):
        self._model = model
        self._idx2word = idx2word

        self._base_unigrams = {}
        self._base_bigrams = {}
        self._base_trigrams = set()
        self._avg_unigram_backoff = 0
        self._avg_bigram_backoff = 0

        self._rnn_ngram_context = rnn_ngram_context
        self._ngram_pruning = ngram_pruning
        self.history_pruning = history_pruning
        self._rnn_ngrams = {}
        for n in range(2, rnn_ngram_context+1):
            self._rnn_ngrams[n] = defaultdict(lambda: [0, 0])

        self._rnn_history = {}
        for n in range(2, rnn_ngram_context):
            self._rnn_history[n] = set()

    def read_base_lm(self, lm_file):
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
                        if line != '\\3-grams:':
                            try:
                                prob, history_1, history_2, word = line.split()
                                trigram = (history_1, history_2,  word)
                                self._base_trigrams.add(trigram)
                            except ValueError:
                                pass
                        else:
                            pass

        self._avg_unigram_backoff /= unigram_cnt
        self._avg_bigram_backoff /= bigram_cnt

    def _update_rnn_ngram_prob_batch(self, n, batch_data, batch_output):
        logging.debug('Processing {}-grams from RNN'.format(n))
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
                    try:
                        if tuple(history) not in self._rnn_history[n-1]:
                            continue
                    except KeyError:
                        pass

                    output_probs = batch_output[sent_id][curr_word_pos]
                    for pred_word_idx, pred_word_log_prob in enumerate(output_probs):
                        if pred_word_idx == PAD_ID or pred_word_idx == BOS_ID:
                            continue
                        pred_word = self._idx2word[pred_word_idx]
                        self._rnn_ngrams[n][tuple(history + [pred_word])][0] += pred_word_log_prob.item()
                        self._rnn_ngrams[n][tuple(history + [pred_word])][1] += 1

    def convert_to_arpa(self, data_iter):
        self._model.eval()
        with torch.no_grad():
            for n in self._rnn_ngrams:
                for batch in tqdm(data_iter):
                    batch_data, batch_targets, batch_pad_lengths = batch[0], batch[1], batch[2]
                    batch_data, batch_targets, batch_pad_lengths = sort_by_lengths(batch_data,
                                                                                   batch_targets,
                                                                                   batch_pad_lengths)
                    batch_output = self._model(batch_data, batch_pad_lengths)

                    self._update_rnn_ngram_prob_batch(n, batch_data, batch_output)
                    del batch_data
                    del batch_targets
                    del batch_pad_lengths
                    del batch_output

                logging.debug('RNN {}-Gram dict size: {}'.format(n, len(self._rnn_ngrams[n])))
                # Average and Normalization
                prob_sum = 0
                for n_gram in self._rnn_ngrams[n]:
                    self._rnn_ngrams[n][n_gram][0] /= self._rnn_ngrams[n][n_gram][1]
                    prob_sum += self._rnn_ngrams[n][n_gram][0]
                for n_gram in self._rnn_ngrams[n]:
                    self._rnn_ngrams[n][n_gram][0] /= prob_sum
                logging.debug('Post-Normalization RNN {}-Gram dict size: {}'.format(n,
                                                                                    len(self._rnn_ngrams[n])))

                # Pruning: Keep only top n NGrams by RNN probability approximation + NGrams in original LM
                try:
                    top_ngrams = set([i[0] for i in sorted(self._rnn_ngrams[n].items(),
                                                           key=lambda x:x[1][0], reverse=True)[:self._ngram_pruning[n]]])
                    if n == 2:
                        top_ngrams.update(set(self._base_bigrams.keys()))
                    elif n == 3:
                        top_ngrams.update(self._base_trigrams)
                    logging.debug('{}-grams to keep after pruning: {}'.format(n, len(top_ngrams)))
                    ngrams_to_remove = set(self._rnn_ngrams[n].keys()).difference(top_ngrams)
                    logging.debug('{}-grams to prune: {}'.format(n, len(ngrams_to_remove)))
                    for ngram in ngrams_to_remove:
                        del self._rnn_ngrams[n][ngram]
                    logging.debug('Post Pruning RNN {}-Gram dict size: {}'.format(n, len(self._rnn_ngrams[n])))
                    del top_ngrams
                    del ngrams_to_remove
                except KeyError:
                    pass

                # History Pruning
                try:
                    history_count = self.history_pruning[n]
                    self._rnn_history[n] = set([i[0] for i in sorted(self._rnn_ngrams[n].items(),
                                                                     key=lambda x: x[1][0],
                                                                     reverse=True)[:history_count]])
                    if n == 2:
                        self._rnn_history[n].update(set(self._base_bigrams.keys()))
                    elif n == 3:
                        self._rnn_history[n].update(self._base_trigrams)
                    logging.debug('{}-grams kept as history for higher n-grams: {}'.format(n,
                                                                                           len(self._rnn_history[n])))
                except KeyError:
                    pass

    def write_arpa_format(self, file_path):
        model_file = codecs.open(file_path, "w+", encoding="utf8")

        print("\\data\\", file=model_file)
        print("ngram 1={0}".format(len(self._base_unigrams)), file=model_file)
        print("ngram 2={0}".format(len(self._rnn_ngrams[2])), file=model_file)
        print("ngram 3={0}".format(len(self._rnn_ngrams[3])), file=model_file)

        print("\n\\1-grams:", file=model_file)
        for unigram in self._base_unigrams:
            if unigram == '<s>' or unigram == '</s>':
                print("{0} {1}".format(self._base_unigrams[unigram].log_prob,
                                       unigram),
                      file=model_file)
            else:
                print("{0} {1} {2}".format(self._base_unigrams[unigram].log_prob,
                                           unigram,
                                           self._base_unigrams[unigram].backoff_weight, file=model_file))

        print("\n\\2-grams:", file=model_file)
        for bigram in self._rnn_ngrams[2]:
            try:
                print("{0} {1} {2} {3}".format(self._rnn_ngrams[2][bigram][0],
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
