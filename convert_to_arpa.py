import argparse
import logging
from lm_models.word import RNNLM
import os
import torch
from utils.arpa import ARPAConverter
from utils.data_reader import DatasetReader
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_base_dir', help='Path to input data dir')
    parser.add_argument('--output_base_dir', help='Path to output base dir')
    parser.add_argument('--base_3gram_lm_path', help='Path to LM file to use as base')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML Config file')
    parser.add_argument('--model_type', default='word', help='RNN model type')
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--tie_weights', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=True, help='Run with DEBUG logging level')

    args = parser.parse_args()
    with open(args.config, 'r') as infile:
        config = yaml.load(infile)

    if args.debug:
        logging.basicConfig(format='%(levelname)s:%(funcName)s:%(lineno)s:\t%(message)s', level=logging.DEBUG)

    data_reader = DatasetReader()
    data_iters = data_reader.get_dataset(data_dir=args.input_base_dir, max_seq_len=config['data']['max_sentence_len'],
                                         batch_size=config['training']['batch_size'], device='cpu')
    args.vocab_size = len(data_reader)

    model = RNNLM(config=config['model'], vocab_size=args.vocab_size,
                  attention=args.attention, tie_weights=args.tie_weights)
    model.load_state_dict(torch.load(args.output_base_dir + '/{}_nnlm.pt'.format(args.model_type), map_location='cpu'))

    arpa_converter = ARPAConverter(idx2word=data_reader.inverted_vocab,
                                   rnn_ngram_context=config['arpa_conversion']['rnn_ngram_context'],
                                   ngram_pruning=config['arpa_conversion']['ngram_pruning'],
                                   history_pruning=config['arpa_conversion']['history_pruning'],
                                   model=model)
    arpa_converter.read_base_lm(args.base_3gram_lm_path)
    arpa_converter.convert_to_arpa(data_iters[0])
    arpa_converter.write_arpa_format(args.output_base_dir +
                                     '/{}_bolm.lm'.format(config['arpa_conversion']['rnn_ngram_context']))


if __name__ == '__main__':
    main()
