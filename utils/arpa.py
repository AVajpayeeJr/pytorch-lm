import codecs
from collections import defaultdict, namedtuple
import logging
import torch
from tqdm import tqdm
from utils.data_reader import PAD_ID, BOS_ID, EOS_ID
from utils.trainer import sort_by_lengths


class ARPAConverter:
    def __init__(self, idx2word, model, rnn_ngram_context=3,
                 ngram_pruning={2: 2000000, 3: 300000}, history_pruning={2:50000}, device='cpu'):
        self._model = model
        self._device = device
        self._idx2word = idx2word

        self._base_unigrams = {}
        self._base_bigrams = {}
        self._base_trigrams = {}
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
                                log_prob, history, word, backoff = line.split()
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
                                log_prob, history_1, history_2, word = line.split()
                                trigram = (history_1, history_2,  word)
                                self._base_trigrams[trigram] = log_prob
                            except ValueError:
                                pass
                        else:
                            pass

        self._avg_unigram_backoff /= unigram_cnt
        self._avg_bigram_backoff /= bigram_cnt

    def _get_rnn_probs(self, n, batch_data, batch_output):
        batch_size = batch_data.shape[0]
        seq_len = batch_data.shape[1]
        vocab_size = batch_output.shape[2]

        # Transforming Batch Data to get n-1 grams
        lower_grams = batch_data.unfold(-1, n-1, 1)
        repeated_lower_grams = torch.repeat_interleave(lower_grams, repeats=vocab_size, dim=1)

        # Transforming to N-Grams
        vocab_indices = torch.arange(vocab_size, device=self._device).repeat(batch_size, seq_len, 1)
        vocab_indices = vocab_indices.narrow(dim=1, start=n-2, length=seq_len-n+2)
        vocab_indices = vocab_indices.view(batch_size, (seq_len-n+2)*vocab_size, 1)
        ngrams = torch.cat((repeated_lower_grams, vocab_indices), dim=-1).float()

        # Adding Probability as last column
        probs = batch_output.narrow(dim=1, start=n-2, length=seq_len-n+2)
        probs = probs.view(batch_size, (seq_len-n+2)*vocab_size, 1)
        ngram_probs = torch.cat((ngrams, probs), dim=-1)
        ngram_probs = ngram_probs.view(batch_size * (seq_len - n + 2) * vocab_size, n + 1)

        # Removing N-Grams ending with PAD_ID
        for i in range(n):
            to_keep = ngram_probs[:, i] != PAD_ID
            ngram_probs = ngram_probs[to_keep]

        # Removing N-Grams with EOS_ID in N-1 Gram
        for i in range(n-1):
            to_keep = ngram_probs[:, i] != EOS_ID
            ngram_probs = ngram_probs[to_keep]

        # Removing N-Grams with BOS_ID as final gram
        to_keep = ngram_probs[:, n-1] != BOS_ID
        ngram_probs = ngram_probs[to_keep]

        return ngram_probs

    def _check_lower_grams(self, n, ngram_probs):
        lower_ngrams = ngram_probs[:, : n].cpu().tolist()

        to_keep = torch.ones(len(lower_ngrams), device=self._device)
        for i, lower_ngram in enumerate(lower_ngrams):
            lower_ngram = tuple([self._idx2word[int(j)] for j in lower_ngram])
            if lower_ngram not in self._rnn_history[n-1]:
                to_keep[i] = 0
        return to_keep

    def _populate_rnn_ngrams(self, n, ngram_probs):
        logging.debug('Populating RNN {}-grams'.format(n))
        ngram_probs = ngram_probs.cpu().numpy()

        for i in tqdm(ngram_probs):
            ngram = tuple([self._idx2word[int(j)] for j in i[:-1]])
            log_prob = i[-1]
            self._rnn_ngrams[n][ngram][0] += log_prob
            self._rnn_ngrams[n][ngram][1] += 1

    def convert_to_arpa(self, data_iter):
        self._model.eval()
        with torch.no_grad():
            for n in range(2, self._rnn_ngram_context + 1):
                logging.debug('Processing {}-grams from RNN'.format(n))
                ngram_probs = torch.Tensor(device=self._device)
                cnt = 0
                for batch in tqdm(data_iter):
                    batch_data, batch_targets, batch_pad_lengths = batch[0], batch[1], batch[2]
                    batch_data, batch_targets, batch_pad_lengths = sort_by_lengths(batch_data,
                                                                                   batch_targets,
                                                                                   batch_pad_lengths)
                    batch_output = self._model(batch_data, batch_pad_lengths)

                    batch_ngram_probs = self._get_rnn_probs(n, batch_data, batch_output)
                    if n > 2:
                        to_keep = self._check_lower_grams(n, batch_ngram_probs)
                        batch_ngram_probs = batch_ngram_probs[to_keep]
                    ngram_probs = torch.cat((ngram_probs, batch_ngram_probs), dim=0)
                    cnt += 1
                    if cnt == 5:
                        break

                logging.debug('NGram Probs Shape: {}'.format(ngram_probs.shape))
                self._populate_rnn_ngrams(n, ngram_probs)
                del ngram_probs

                logging.debug('RNN {}-Gram dict size: {}'.format(n, len(self._rnn_ngrams[n])))

                logging.debug('Averaging RNN {}-grams'.format(n))
                prob_sum = 0
                for n_gram in tqdm(self._rnn_ngrams[n]):
                    self._rnn_ngrams[n][n_gram][0] /= self._rnn_ngrams[n][n_gram][1]
                    prob_sum += self._rnn_ngrams[n][n_gram][0]
                logging.debug('Normalizing RNN {}-grams'.format(n))
                for n_gram in tqdm(self._rnn_ngrams[n]):
                    self._rnn_ngrams[n][n_gram][0] /= prob_sum
                logging.debug('Post-Normalization RNN {}-Gram dict size: {}'.format(n,
                                                                                    len(self._rnn_ngrams[n])))

                # Pruning: Keep only top n NGrams by RNN probability approximation + NGrams in original LM
                try:
                    logging.debug('Pruning RNN {}-grams'.format(n))
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
                    logging.debug('Populating RNN {}-grams to keep as history for higher grams'.format(n))
                    history_count = self.history_pruning[n]
                    self._rnn_history[n] = set([i[0] for i in sorted(self._rnn_ngrams[n].items(),
                                                                     key=lambda x: x[1][0],
                                                                     reverse=True)[:history_count]])
                    if n == 2:
                        self._rnn_history[n].update(set(self._base_bigrams.keys()))
                    elif n == 3:
                        self._rnn_history[n].update(set(self._base_trigrams.keys()))
                    logging.debug('{}-grams kept as history for higher n-grams: {}'.format(n,
                                                                                           len(self._rnn_history[n])))
                except KeyError:
                    pass

    def write_arpa_format(self, file_path):
        model_file = codecs.open(file_path, "w+", encoding="utf8")

        print("\\data\\", file=model_file)
        print("ngram 1={0}".format(len(self._base_unigrams)), file=model_file)
        print("ngram 2={0}".format(len(self._rnn_ngrams[2])), file=model_file)
        try:
            print("ngram 3={0}".format(len(self._rnn_ngrams[3])), file=model_file)
        except KeyError:
            print("ngram 3={0}".format(len(self._base_trigrams)), file=model_file)

        print("\n\\1-grams:", file=model_file)
        for unigram in self._base_unigrams:
            if unigram == '<s>' or unigram == '</s>':
                print("{0} {1}".format(self._base_unigrams[unigram].log_prob,
                                       unigram),
                      file=model_file)
            else:
                print("{0} {1} {2}".format(self._base_unigrams[unigram].log_prob,
                                           unigram,
                                           self._base_unigrams[unigram].backoff_weight),
                      file=model_file)

        print("\n\\2-grams:", file=model_file)
        for bigram in self._rnn_ngrams[2]:
            try:
                print("{0} {1} {2} {3}".format(self._rnn_ngrams[2][bigram][0],
                                               bigram[0],
                                               bigram[1],
                                               self._base_bigrams[bigram].backoff_weight),
                      file=model_file)
            except ValueError:
                print("{0} {1} {2}".format(-99,
                                           bigram[0],
                                           bigram[1],
                                           self._base_bigrams[bigram].backoff_weight),
                      file=model_file)
            except KeyError:
                print("{0} {1} {2}".format(self._rnn_ngrams[2][bigram][0],
                                           bigram[0],
                                           bigram[1],
                                           self._avg_bigram_backoff),
                      file=model_file)

        print("\n\\3-grams:", file=model_file)
        try:
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
        except KeyError:
            for trigram in self._base_trigrams:
                try:
                    print("{0} {1} {2} {3}".format(self._base_trigrams[trigram],
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
