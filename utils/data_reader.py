import codecs
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PAD = '<PAD>'
_EOS = '<eos>'
_UNK = '<unk>'

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

START_VOCAB = [_PAD, _EOS, _UNK]


class DatasetReader:
    def __init__(self):
        self._tokens2idx = dict(zip(START_VOCAB, [i for i in range(len(START_VOCAB))]))
        self._idx2tokens = dict(zip([i for i in range(len(START_VOCAB))], START_VOCAB))

    def _build_vocab(self, sentences):
        idx_cnt = 3
        for sent in sentences:
            for word in sent:
                if word in self._tokens2idx:
                    continue
                else:
                    self._tokens2idx[word] = idx_cnt
                    idx_cnt += 1
        self._idx2tokens = {v: k for k, v in self._tokens2idx.items()}

    @staticmethod
    def _read_file(filename):
        with codecs.open(filename, 'r', encoding='utf-8') as infile:
            sentences = [j.strip().split() + [_EOS] for j in infile.readlines() if j.strip()]
        return sentences

    def _sentences_to_token_ids(self, sentences):
        token_ids = []
        for sent in sentences:
            token_ids.append([self._tokens2idx.get(w, UNK_ID) for w in sent])
        return token_ids

    def _read_data(self, data_dir):
        train_path = data_dir + '/train.txt'
        valid_path = data_dir + '/valid.txt'
        test_path = data_dir + '/test.txt'

        train_sentences = self._read_file(train_path)
        self._build_vocab(train_sentences)
        train_token_ids = self._sentences_to_token_ids(train_sentences)

        valid_sentences = self._read_file(valid_path)
        valid_token_ids = self._sentences_to_token_ids(valid_sentences)

        test_sentences = self._read_file(test_path)
        test_token_ids = self._sentences_to_token_ids(test_sentences)

        return train_token_ids, valid_token_ids, test_token_ids

    @staticmethod
    def _lm_data_producer(data, max_seq_len, dtype=np.long):
        padded_data = []
        lengths = []

        for seq in data:
            lengths.append(min(len(seq), max_seq_len))
            if len(seq) < max_seq_len + 1:
                padded_data.append(seq + [PAD_ID] * (max_seq_len + 1 - len(seq)))
            else:
                padded_data.append(seq[0:(max_seq_len + 1)])

        padded_data = np.array(padded_data, dtype=dtype)
        lengths = np.array(lengths, dtype=dtype)

        xtrain = padded_data[:, 0: max_seq_len].astype(dtype)
        ytrain = padded_data[:, 1: max_seq_len + 1].astype(dtype)

        return xtrain, ytrain, lengths

    def get_dataset(self, data_dir, max_seq_len, batch_size, device):
        train_data, valid_data, test_data = self._read_data(data_dir)

        train_data = self._lm_data_producer(train_data, max_seq_len)
        train_x = torch.tensor(train_data[0], dtype=torch.long, device=device)
        train_y = torch.tensor(train_data[1], dtype=torch.long, device=device)
        train_lengths = torch.tensor(train_data[2], dtype=torch.float, device=device)
        train_dataset = TensorDataset(train_x, train_y, train_lengths)

        valid_data = self._lm_data_producer(valid_data, max_seq_len)
        valid_x = torch.tensor(valid_data[0], dtype=torch.long, device=device)
        valid_y = torch.tensor(valid_data[1], dtype=torch.long, device=device)
        valid_lengths = torch.tensor(valid_data[2], dtype=torch.float, device=device)
        valid_dataset = TensorDataset(valid_x, valid_y, valid_lengths)

        test_data = self._lm_data_producer(test_data, max_seq_len)
        test_x = torch.tensor(test_data[0], dtype=torch.long, device=device)
        test_y = torch.tensor(test_data[1], dtype=torch.long, device=device)
        test_lengths = torch.tensor(test_data[2], dtype=torch.float, device=device)
        test_dataset = TensorDataset(test_x, test_y, test_lengths)

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
