import argparse
import logging
from lm_models.lm import RNNLM
import torch
from utils.data_reader import DatasetReader
from utils.trainer import Trainer
import yaml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 123


def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN Language Modeling')
    parser.add_argument('--output_base_dir', help='Path to output base directory')
    parser.add_argument('--input_base_dir', help='Path to input directory')
    parser.add_argument('--config', help='Path to YAML Config file')
    parser.add_argument('--model_type', default='word', help='word/char/word_char')
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=True, help='Run with DEBUG logging level')

    args = parser.parse_args()
    with open(args.config, 'r') as infile:
        config = yaml.load(infile)

    save_dir = args.output_base_dir + '/' + \
               '/neural/{}_{}_attention={}'.format(args.model_type,
                                                                  config['model']['encoder']['type'],
                                                                  args.attention)
    if args.debug:
        logging.basicConfig(format='%(levelname)s:%(funcName)s:%(lineno)s:\t%(message)s', level=logging.DEBUG)

    data_reader = DatasetReader(mode=args.model_type)
    data_iters = data_reader.get_dataset(data_dir=args.input_base_dir,
                                         max_seq_len=config['data']['max_sentence_len'],
                                         max_word_len=config['data']['max_word_len'],
                                         batch_size=config['training']['batch_size'], device=DEVICE)

    train_iter, val_iter, test_iter = data_iters[0], data_iters[1], data_iters[2]
    logging.info('Train Batches: {}'.format(len(train_iter)))
    logging.info('Valid Batches: {}'.format(len(val_iter)))
    logging.info('Test Batches: {}'.format(len(test_iter)))

    args.device = DEVICE
    args.word_vocab_size = len(data_reader.vocab)
    args.char_vocab_size = len(data_reader.char_vocab)
    args.save_dir = save_dir
    args.file_name = args.model_type + '_nnlm'

    model = RNNLM(config=config, word_vocab_size=args.word_vocab_size,
                  char_vocab_size=args.char_vocab_size, mode=args.model_type,
                  attention=args.attention)
    model.to(DEVICE)
    trainer = Trainer(args=args, config=config['training'], model=model,
                      train_iter=train_iter, val_iter=val_iter, test_iter=test_iter)
    trainer.train()


if __name__ == '__main__':
    main()
