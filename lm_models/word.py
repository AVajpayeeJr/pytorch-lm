from lm_models.attention import Attention
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNLM(nn.Module):
    def __init__(self, vocab_size, config, tie_weights=True, attention=True):
        super().__init__()

        word_embedding_size = config['embedding']['word']['dim']
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_size)

        encoder_hidden_size = config['encoder']['hidden_size']
        if config['encoder']['type'] == 'LSTM':
            self.encoder = nn.LSTM(input_size=word_embedding_size,
                                   hidden_size=encoder_hidden_size,
                                   num_layers=config['encoder']['num_layers'],
                                   batch_first=True,
                                   dropout=config['encoder']['dropout'])
        elif config['encoder']['type'] == 'GRU':
            self.encoder = nn.GRU(input_size=word_embedding_size,
                                  hidden_size=encoder_hidden_size,
                                  num_layers=config['encoder']['num_layers'],
                                  batch_first=True,
                                  dropout=config['encoder']['dropout'])

        self.attention = attention
        if self.attention:
            self.attention_score_module = Attention(encoder_hidden_size)
            self.concatenation_layer = nn.Linear(encoder_hidden_size * 2, encoder_hidden_size)

        self.decoder = nn.Linear(in_features=encoder_hidden_size,
                                 out_features=vocab_size)

        if tie_weights:
            if word_embedding_size != encoder_hidden_size:
                raise ValueError(
                    'When using the tied flag, encoder embedding_size must be equal to hidden_size')
            self.decoder.weight = self.word_embed.weight

        self.output = nn.LogSoftmax(dim=-1)
        self.init_weights()

    def forward(self, word_input, word_pad_lengths):
        total_length = word_input.size(1)  # get the max sequence length

        embedded = self.word_embed(word_input)

        self.flatten_parameters()

        packed_input = pack_padded_sequence(embedded, word_pad_lengths, batch_first=True)
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
        self.word_embed.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
