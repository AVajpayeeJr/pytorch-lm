from lm_models.attention import Attention
from lm_models.embedding import TokenEmbedding
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNLM(nn.Module):
    def __init__(self, config, word_vocab_size, char_vocab_size=None, mode='word', attention=True):
        super().__init__()

        self.mode = mode
        self.embedding_layer = TokenEmbedding(config=config,
                                              word_vocab_size=word_vocab_size,
                                              char_vocab_size=char_vocab_size)
        self.embedding_size = self.embedding_layer.embedding_size

        encoder_hidden_size = config['model']['encoder']['hidden_size']
        if config['model']['encoder']['type'] == 'LSTM':
            self.encoder = nn.LSTM(input_size=self.embedding_size,
                                   hidden_size=encoder_hidden_size,
                                   num_layers=config['model']['encoder']['num_layers'],
                                   batch_first=True,
                                   dropout=config['model']['encoder']['dropout'])
        elif config['model']['encoder']['type'] == 'GRU':
            self.encoder = nn.GRU(input_size=self.embedding_size,
                                  hidden_size=encoder_hidden_size,
                                  num_layers=config['model']['encoder']['num_layers'],
                                  batch_first=True,
                                  dropout=config['model']['encoder']['dropout'])

        self.attention = attention
        if self.attention:
            self.attention_score_module = Attention(encoder_hidden_size)
            self.concatenation_layer = nn.Linear(encoder_hidden_size * 2, encoder_hidden_size)

        self.decoder = nn.Linear(in_features=encoder_hidden_size,
                                 out_features=word_vocab_size)

        self.output = nn.LogSoftmax(dim=-1)
        self.init_weights()

    def forward(self, seq_pad_lengths, word_input, word_char_input=None):
        total_length = word_input.size(1)  # get the max sequence length

        embedded = self.embedding_layer(word_input, word_char_input)

        self.flatten_parameters()

        packed_input = pack_padded_sequence(embedded, seq_pad_lengths, batch_first=True)
        packed_output, _ = self.encoder(packed_input)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)

        if self.attention:
            context_vectors, attention_score = self.attention_score_module(encoder_output)
            combined_encoding = torch.cat((context_vectors, encoder_output), dim=2)
            encoder_output = torch.tanh(self.concatenation_layer(combined_encoding))

        decoded = self.decoder(encoder_output)

        probs = self.output(decoded)
        return probs

    def flatten_parameters(self):
        self.encoder.flatten_parameters()

    def init_weights(self, init_range=0.1):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
