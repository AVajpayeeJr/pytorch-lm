import argparse
import logging
from lm_models.word import RNNLM
import torch
from utils.arpa import ARPAConverter
from utils.data_reader import DatasetReader
from utils.trainer import Trainer
import yaml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 123


def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN Language Modeling')
    parser.add_argument('--output_base_dir', help='Path to output base directory')
    parser.add_argument('--language', help='Language code')
    parser.add_argument('--config', help='Path to YAML Config file')
    parser.add_argument('--model_type', help='word')
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--tie_weights', action='store_true', default=False)
    parser.add_argument('--input_ngram_lm', default=None, help='Input NGram LM file to use as base for approximating.')
    parser.add_argument('--output_ngram_lm', default=None, help='Output NGram LM file to write (SRILM Format)')
    parser.add_argument('--debug', action='store_true', default=True, help='Run with DEBUG logging level')

    args = parser.parse_args()
    with open(args.config, 'r') as infile:
        config = yaml.load(infile)

    save_dir = args.output_base_dir + '/' + args.language + \
               '/neural/{}_{}_attention={}_tie-weights={}'.format(args.model_type,
                                                                  config['model']['encoder']['type'],
                                                                  args.attention,
                                                                  args.tie_weights)
    if not args.input_ngram_lm:
        args.input_ngram_lm = args.output_base_dir + '/' + args.language + '/ngram/3gram_kn_interp.lm'
    if not args.output_ngram_lm:
        args.output_ngram_lm = save_dir + '/' + args.language + '_' + args.model_type + '_' + 'neural_arpa.lm'

    if args.debug:
        logging.basicConfig(format='%(levelname)s:%(funcName)s:%(lineno)s:\t%(message)s', level=logging.DEBUG)

    data_reader = DatasetReader()
    data_iters = data_reader.get_dataset(data_dir=args.language, max_seq_len=config['data']['max_sentence_len'],
                                         batch_size=config['training']['batch_size'], device=DEVICE)

    train_iter, val_iter, test_iter = data_iters[0], data_iters[1], data_iters[2]
    logging.info('Train Batches: {}'.format(len(train_iter)))
    logging.info('Valid Batches: {}'.format(len(val_iter)))
    logging.info('Test Batches: {}'.format(len(test_iter)))

    args.vocab_size = len(data_reader)
    args.save_dir = save_dir
    args.file_name = args.model_type + '_nnlm'

    arpa_converter = ARPAConverter(word2idx=data_reader.vocab)
    arpa_converter.read_lm(args.input_ngram_lm)

    model = RNNLM(config=config['model'], vocab_size=args.vocab_size,
                  attention=args.attention, tie_weights=args.tie_weights)
    model.to(DEVICE)
    trainer = Trainer(args=args, config=config['training'], model=model,
                      train_iter=train_iter, val_iter=val_iter, test_iter=test_iter,
                      arpa_converter=arpa_converter)
    trainer.train()


if __name__ == '__main__':
    main()
