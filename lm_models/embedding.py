from lm_models.attention import Attention
import torch
import torch.nn as nn
from utils.data_reader import PAD_ID


class CharRNNEmbedding(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        self.char_embedding_size = config['model']['embedding']['char']['dim']
        self.char_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=self.char_embedding_size,
                                           padding_idx=PAD_ID)
        self._init_char_embedding(padding_idx=PAD_ID)

        self.rnn_type = config['model']['embedding']['char']['type']
        self.char_rnn_hidden_size = config['model']['embedding']['char'][self.rnn_type]['hidden_size']
        self.char_rnn_layers = config['model']['embedding']['char'][self.rnn_type]['num_layers']
        self.char_rnn_dropout = config['model']['embedding']['char'][self.rnn_type]['dropout']

        if self.rnn_type == 'LSTM':
            self.char_rnn_embedding = nn.LSTM(input_size=self.char_embedding_size,
                                              hidden_size=self.char_rnn_hidden_size,
                                              num_layers=self.char_rnn_layers,
                                              dropout=self.char_rnn_dropout,
                                              batch_first=False,
                                              bidirectional=True)
        elif self.rnn_type == 'GRU':
            self.char_rnn_embedding = nn.GRU(input_size=self.char_embedding_size,
                                             hidden_size=self.char_rnn_hidden_size,
                                             num_layers=self.char_rnn_layers,
                                             dropout=self.char_rnn_dropout,
                                             batch_first=False,
                                             bidirectional=True)
        else:
            raise KeyError('Cannot create CharRNNEmbedding with type {}'.format(self.rnn_type))

        self.char_linear_embedding = nn.Linear(in_features=2 * self.char_rnn_hidden_size,
                                               out_features=self.char_rnn_hidden_size)

        self.embedding_size = self.char_rnn_hidden_size
        self._init_linear_weights_and_bias()

    def _init_char_embedding(self, padding_idx):
        nn.init.xavier_normal_(self.char_embedding.weight)
        self.char_embedding.weight.data[padding_idx].uniform_(0, 0)

    def _init_rnn_weights(self):
        for idx in range(len(self.char_rnn_embedding.all_weights[0])):
            dim = self.char_rnn_embedding.all_weights[0][idx].size()
            if len(dim) < 2:
                nn.init.constant_(self.char_rnn_embedding.all_weights[0][idx], 1)
            elif len(dim) == 2:
                nn.init.xavier_uniform_(self.char_rnn_embedding.all_weights[0][idx])

    def _init_linear_weights_and_bias(self):
        # Init linear weights
        nn.init.xavier_uniform_(self.char_linear_embedding.weight)
        # Init bias weights
        nn.init.constant_(self.char_linear_embedding.bias, 1)

    def forward(self, char_word_seq_input):
        char_embeddings = []
        seq_len = char_word_seq_input.size(1)
        for i in range(seq_len):
            x = char_word_seq_input[:, i, :]
            x = self.char_embedding(x)
            x = x.transpose(0, 1)
            char_embedding, _ = self.char_rnn_embedding(x)
            char_embedding = char_embedding.transpose(0, 1)
            char_embedding = torch.cat(
                [
                    char_embedding[:, 0, :self.char_rnn_hidden_size],
                    char_embedding[:, -1, self.char_rnn_hidden_size:]
                ],
                dim=1)
            char_embedding = self.char_linear_embedding(char_embedding)
            char_embedding = char_embedding.unsqueeze(1)
            char_embeddings.append(char_embedding)

        # Concatenate the whole char embeddings
        final_char_embedding = torch.cat(char_embeddings, dim=1)

        return final_char_embedding


class WordEmbedding(nn.Module):
    def __init__(self, config, vocab_size):
        super(WordEmbedding, self).__init__()
        self.embedding_size = config['model']['embedding']['word']['dim']
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=self.embedding_size,
                                           padding_idx=PAD_ID)

    def forward(self, x_word):
        return self.word_embedding(x_word)


class TokenEmbedding(nn.Module):
    def __init__(self, config, word_vocab_size=None, char_vocab_size=None, mode='word'):
        super(TokenEmbedding, self).__init__()

        self.mode = mode
        self.embedding_size = 0
        if self.mode == 'word' or self.mode == 'word_char':
            self.word_embedding_layer = WordEmbedding(config=config, vocab_size=word_vocab_size)
            self.embedding_size += self.word_embedding_layer.embedding_size

        if self.mode == 'char' or self.mode == 'word_char':
            self.char_embedding_layer = CharRNNEmbedding(config=config, vocab_size=char_vocab_size)
            self.embedding_size += self.char_embedding_layer.embedding_size

    def forward(self, x_word, x_char):
        if self.mode == 'word':
            return self.word_embedding_layer(x_word)
        elif self.mode == 'char':
            return self.char_embedding_layer(x_char)
        elif self.mode == 'word_char':
            word_embed, word_embed_dim = self.word_embedding_layer(x_word)
            char_embed, char_embed_dim = self.char_embedding_layer(x_char)
            final_embed = torch.cat([word_embed, char_embed], 2)
            return final_embed
