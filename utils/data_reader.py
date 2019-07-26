import codecs
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset

_PAD = '<PAD>'
_BOS = '<s>'
_EOS = '</s>'
_UNK = '<unk>'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

START_VOCAB = [_PAD, _BOS, _EOS, _UNK]


class DatasetReader:
    def __init__(self, mode='word'):
        self._mode = mode

        self._tokens2idx = dict(zip(START_VOCAB, [i for i in range(len(START_VOCAB))]))
        self._idx2tokens = dict(zip([i for i in range(len(START_VOCAB))], START_VOCAB))

        self._chars2idx = dict(zip(START_VOCAB, [i for i in range(len(START_VOCAB))]))
        self._idx2chars = dict(zip([i for i in range(len(START_VOCAB))], START_VOCAB))

    def _build_vocab(self, sentences):
        idx_cnt = 4
        char_idx_cnt = 4
        for sent in sentences:
            for word in sent:
                if word in self._tokens2idx:
                    pass
                else:
                    self._tokens2idx[word] = idx_cnt
                    idx_cnt += 1
                if self._mode == 'char' or self._mode == 'word_char':
                    for char in word:
                        if char in self._chars2idx:
                            pass
                        else:
                            self._chars2idx[char] = char_idx_cnt
                            char_idx_cnt += 1

        self._idx2tokens = {v: k for k, v in self._tokens2idx.items()}
        self._idx2chars = {v: k for k, v in self._chars2idx.items()}

    @staticmethod
    def _read_file(filename):
        with codecs.open(filename, 'r', encoding='utf-8') as infile:
            sentences = [j.strip().split() for j in infile.readlines() if j.strip()]
        return sentences

    def _sentences_to_ids(self, sentences):
        token_ids_list = []
        char_ids_list = []
        for sent in sentences:
            token_ids_list.append([BOS_ID] + [self._tokens2idx.get(w, UNK_ID) for w in sent] + [EOS_ID])
            if self._mode == 'char' or self._mode == 'word_char':
                char_ids_list.append([[BOS_ID]] +
                                     [[self._chars2idx.get(c, UNK_ID) for c in w] for w in sent] +
                                     [[EOS_ID]])

        return token_ids_list, char_ids_list

    def _read_data(self, data_dir):
        train_path = data_dir + '/train.txt'
        valid_path = data_dir + '/valid.txt'
        test_path = data_dir + '/test.txt'

        train_sentences = self._read_file(train_path)
        self._build_vocab(train_sentences)
        train_ids = self._sentences_to_ids(train_sentences)

        valid_sentences = self._read_file(valid_path)
        valid_ids = self._sentences_to_ids(valid_sentences)

        test_sentences = self._read_file(test_path)
        test_ids = self._sentences_to_ids(test_sentences)

        return train_ids, valid_ids, test_ids

    def _lm_data_producer(self, data, max_seq_len, max_word_len, dtype=np.long):
        word_padded_data = []
        char_padded_data = []
        lengths = []

        data_size = 0
        if self._mode == 'word' or self._mode == 'word_char':
            data_size = len(data[0])
        elif self._mode == 'char':
            data_size = len(data[0])

        for i in range(data_size):
            word_seq = data[0][i]
            lengths.append(min(len(word_seq), max_seq_len))
            if len(word_seq) < max_seq_len + 1:
                word_padded_data.append(word_seq + [PAD_ID] * (max_seq_len + 1 - len(word_seq)))
            else:
                word_padded_data.append(word_seq[0:(max_seq_len + 1)])

            if self._mode == 'char' or self._mode == 'word_char':
                char_sent_seq = []

                for char_word_seq in data[1][i]:
                    if len(char_word_seq) < max_word_len + 1:
                        char_sent_seq.append(char_word_seq + [PAD_ID] * (max_word_len + 1 - len(char_word_seq)))
                    else:
                        char_sent_seq.append(char_word_seq[0:(max_word_len + 1)])

                if len(word_seq) < max_seq_len + 1:
                    char_padded_data.append(char_sent_seq + [[PAD_ID] * (max_word_len + 1)]
                                            * (max_seq_len + 1 - len(char_sent_seq)))
                else:
                    char_padded_data.append(char_sent_seq[0:(max_seq_len + 1)])

        word_padded_data = np.array(word_padded_data, dtype=dtype)
        #print(word_padded_data)
        lengths = np.array(lengths, dtype=dtype)
        word_x_train = word_padded_data[:, 0: max_seq_len].astype(dtype)
        char_x_train = None
        if self._mode == 'char' or self._mode == 'word_char':
            char_padded_data = np.array(char_padded_data, dtype=dtype)
            char_x_train = char_padded_data[:, 0: max_seq_len].astype(dtype)
        ytrain = word_padded_data[:, 1: max_seq_len + 1].astype(dtype)

        return (word_x_train, char_x_train), ytrain, lengths

    def get_dataset(self, data_dir, max_seq_len, batch_size, device, max_word_len=None):
        if max_word_len is None:
            max_word_len = max_seq_len

        train_data, valid_data, test_data = self._read_data(data_dir)
        train_dataset, valid_dataset, test_dataset = None, None, None

        train_data = self._lm_data_producer(train_data, max_seq_len, max_word_len)
        word_train_x = torch.tensor(train_data[0][0], dtype=torch.long, device=device)
        train_y = torch.tensor(train_data[1], dtype=torch.long, device=device)
        train_lengths = torch.tensor(train_data[2], dtype=torch.float, device=device)
        if self._mode == 'word':
            logging.debug('Word Train X: {} {}'.format(word_train_x.shape, word_train_x.dtype))
            logging.debug('Train Y: {} {}'.format(train_y.shape, train_y.dtype))
            logging.debug('Train Lengths: {} {}'.format(train_lengths.shape, train_lengths.dtype))
            train_dataset = TensorDataset(word_train_x, train_y, train_lengths)
        elif self._mode == 'word_char' or self._mode == 'char':
            char_train_x = torch.tensor(train_data[0][1], dtype=torch.long, device=device)
            logging.debug('Word Train X: {} {}'.format(word_train_x.shape, word_train_x.dtype))
            logging.debug('Char Train X: {} {}'.format(char_train_x.shape, char_train_x.dtype))
            logging.debug('Train Y: {} {}'.format(train_y.shape, train_y.dtype))
            logging.debug('Train Lengths: {} {}'.format(train_lengths.shape, train_lengths.dtype))
            train_dataset = TensorDataset(word_train_x, char_train_x, train_y, train_lengths)

        valid_data = self._lm_data_producer(valid_data, max_seq_len, max_word_len)
        word_valid_x = torch.tensor(valid_data[0][0], dtype=torch.long, device=device)
        valid_y = torch.tensor(valid_data[1], dtype=torch.long, device=device)
        valid_lengths = torch.tensor(valid_data[2], dtype=torch.float, device=device)
        if self._mode == 'word':
            logging.debug('Word Valid X: {} {}'.format(word_valid_x.shape, word_valid_x.dtype))
            logging.debug('Valid Y: {} {}'.format(valid_y.shape, valid_y.dtype))
            logging.debug('Valid Lengths: {} {}'.format(valid_lengths.shape, valid_lengths.dtype))
            valid_dataset = TensorDataset(word_valid_x, valid_y, valid_lengths)
        elif self._mode == 'word_char' or self._mode == 'char':
            char_valid_x = torch.tensor(valid_data[0][1], dtype=torch.long, device=device)
            logging.debug('Word Valid X: {} {}'.format(word_valid_x.shape, word_valid_x.dtype))
            logging.debug('Char Valid X: {} {}'.format(char_valid_x.shape, char_valid_x.dtype))
            logging.debug('Valid Y: {} {}'.format(valid_y.shape, valid_y.dtype))
            logging.debug('Valid Lengths: {} {}'.format(valid_lengths.shape, valid_lengths.dtype))
            valid_dataset = TensorDataset(word_valid_x, char_valid_x, valid_y, valid_lengths)

        test_data = self._lm_data_producer(test_data, max_seq_len, max_word_len)
        word_test_x = torch.tensor(test_data[0][0], dtype=torch.long, device=device)
        test_y = torch.tensor(test_data[1], dtype=torch.long, device=device)
        test_lengths = torch.tensor(test_data[2], dtype=torch.float, device=device)
        if self._mode == 'word':
            logging.debug('Word Test X: {} {}'.format(word_test_x.shape, word_test_x.dtype))
            logging.debug('Test Y: {} {}'.format(test_y.shape, test_y.dtype))
            logging.debug('Test Lengths: {} {}'.format(test_lengths.shape, test_lengths.dtype))
            test_dataset = TensorDataset(word_test_x, test_y, test_lengths)
        elif self._mode == 'word_char' or self._mode == 'char':
            char_test_x = torch.tensor(test_data[0][1], dtype=torch.long, device=device)
            logging.debug('Word Test X: {} {}'.format(word_test_x.shape, word_test_x.dtype))
            logging.debug('Char Test X: {} {}'.format(char_test_x.shape, char_test_x.dtype))
            logging.debug('Test Y: {} {}'.format(test_y.shape, test_y.dtype))
            logging.debug('Test Lengths: {} {}'.format(test_lengths.shape, test_lengths.dtype))
            test_dataset = TensorDataset(word_test_x, char_test_x, test_y, test_lengths)

        train_iter = DataLoader(train_dataset, batch_size=batch_size)
        valid_iter = DataLoader(valid_dataset, batch_size=batch_size)
        test_iter = DataLoader(test_dataset, batch_size=batch_size)

        return train_iter, valid_iter, test_iter

    def __len__(self):
        return len(self._tokens2idx)

    @property
    def vocab(self):
        return self._tokens2idx

    @vocab.setter
    def vocab(self, x):
        self._tokens2idx = x

    @property
    def inverted_vocab(self):
        return self._idx2tokens

    @inverted_vocab.setter
    def inverted_vocab(self, x):
        self._idx2tokens = x

    @property
    def char_vocab(self):
        return self._chars2idx

    @char_vocab.setter
    def char_vocab(self, x):
        self._chars2idx = x

    @property
    def inverted_char_vocab(self):
        return self._idx2chars

    @inverted_char_vocab.setter
    def inverted_char_vocab(self, x):
        self._idx2chars = x