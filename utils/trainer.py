import math
import logging
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, config, model, train_iter, val_iter, test_iter, arpa_converter):
        self.args = args
        self.config = config
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.writer = SummaryWriter(self.args.save_dir + '/runs/' + self.args.file_name)
        self.optimizer = self._get_optimizer()
        self.arpa_converter = arpa_converter

    def _get_optimizer(self):
        if self.config['optimizer']['name'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config['optimizer']['lr'],
                              betas=(0.0, 0.999), eps=1e-8,
                              weight_decay=12e-7, amsgrad=True)
        else:
            raise KeyError('Optimizer {} not implemented.'.format(self.config['optimizer']['name']))

    @staticmethod
    def _sort_by_lengths(data, targets, pad_lengths):
        sorted_pad_lengths, sorted_index = pad_lengths.sort(descending=True)
        data = data[sorted_index, :]
        targets = targets[sorted_index, :]
        return data, targets, sorted_pad_lengths

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.
        total_num_examples = 0
        start_time = time.time()
        iteration_step = len(self.train_iter) * (epoch - 1)

        t = tqdm(iter(self.train_iter), leave=False, total=len(self.train_iter))
        for i, batch in enumerate(t):

            iteration_step += 1

            data, targets, pad_lengths = batch[0], batch[1], batch[2]
            data, targets, pad_lengths = self._sort_by_lengths(data, targets, pad_lengths)

            self.model.zero_grad()
            output = self.model(data, pad_lengths)
            loss = torch.nn.NLLLoss(ignore_index=0, reduction='sum')(output.view(-1, self.args.vocab_size),
                                                                     targets.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            total_num_examples += len(data.nonzero())

            if iteration_step % self.config['log_interval'] == 0 and i > 0:
                cur_loss = total_loss / total_num_examples
                elapsed = time.time() - start_time
                exp_cur_loss = min(cur_loss, 7)

                update_message = 'ms/batch:{} loss:{} ppl:{}'.format(round(elapsed * 1000 / self.config['log_interval'],
                                                                           2),
                                                                     round(cur_loss, 2),
                                                                     round(min(math.exp(exp_cur_loss), 1000), 2))
                t.set_postfix_str(update_message, refresh=False)
                t.update(n=self.config['log_interval'])
                self.writer.add_scalar('training_loss', cur_loss, iteration_step)
                self.writer.add_scalar('training_perplexity',
                                       min(math.exp(exp_cur_loss), 1000), iteration_step)
                total_loss = 0
                total_num_examples = 0
                start_time = time.time()

    def _evaluate(self, data_iterator):
        self.model.eval()
        total_loss = 0.
        word_count = 0

        with torch.no_grad():
            for _, batch in enumerate(tqdm(data_iterator)):
                data, targets, pad_lengths = batch[0], batch[1], batch[2]
                data, targets, pad_lengths = self._sort_by_lengths(data, targets, pad_lengths)
                output = self.model(data, pad_lengths)
                output_flat = output.view(-1, self.args.vocab_size)
                total_loss += torch.nn.NLLLoss(ignore_index=0, reduction='sum')(output_flat,
                                                                                targets.view(-1)).item()
                word_count += len(data.nonzero())

        self.model.train()
        return total_loss / word_count

    @staticmethod
    def _time_string(time_epoch):
        return time.strftime('%m/%d/Y, %H:%M:%S', time.localtime(time_epoch))

    def train(self):
        best_val_loss = False
        for epoch in range(1, self.config['epochs'] + 1):
            train_epoch_start_time = time.time()
            logging.debug('Training Epoch: {}\tStart Time: {}'.format(epoch,
                                                                      self._time_string(train_epoch_start_time)))

            self._train_epoch(epoch=epoch)
            train_epoch_end_time = time.time()
            train_epoch_time = train_epoch_end_time - train_epoch_start_time
            logging.debug('Epoch: {}\tEnd Time: {}\tTime to Train: {}sec'.format(epoch,
                                                                                 self._time_string(train_epoch_end_time),
                                                                                 round(train_epoch_time)))
            logging.debug('Evaluating on Val')
            val_loss = self._evaluate(self.val_iter)
            val_ppl = min(math.exp(val_loss), 1000)
            logging.debug('Evaluating on Test')
            test_loss = self._evaluate(self.test_iter)
            test_ppl = min(math.exp(test_loss), 1000)
            logging.info('Epoch: {}\tVal Loss: {}\tVal PPL: {}\tTest Loss: {}\tTest PPL: {}'.format(epoch,
                                                                                                    round(val_loss, 3),
                                                                                                    round(val_ppl, 3),
                                                                                                    round(test_loss, 3),
                                                                                                    round(test_ppl, 3)
                                                                                                    ))
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('validation_loss_at_epoch', val_loss, epoch)
            self.writer.add_scalar('test_loss_at_epoch', test_loss, epoch)
            self.writer.add_scalar('validation_perplexity_at_epoch', val_ppl, epoch)
            self.writer.add_scalar('test_perplexity_at_epoch', test_ppl, epoch)

            if not best_val_loss or val_loss < best_val_loss:
                if not os.path.exists(self.args.save_dir + '/'):
                    os.makedirs(self.args.save_dir)

                with open('{}/{}.pt'.format(self.args.save_dir, self.args.file_name), 'wb') as f:
                    torch.save(self.model.state_dict(), f)

                self.convert_to_arpa()
                best_val_loss = val_loss

    @staticmethod
    def _batch_label_probs(targets, output):
        y_pred = output.cpu().numpy()
        y_true = targets.cpu().numpy()
        label_probabilities = []
        for sent_id, sent in enumerate(y_true):
            sent_prob_list = []
            for word_pos, word_idx in enumerate(sent):
                if word_idx == 0:
                    # EOS
                    label_probabilities.append(sent_prob_list.copy())
                    break
                else:
                    prob = np.exp(y_pred[sent_id][word_pos][word_idx])
                    sent_prob_list.append((word_idx, prob))
            sent_prob_list.clear()
        return label_probabilities

    def _get_label_probabilities(self):
        logging.debug('Getting label_probabilities')
        label_probabilities = []
        with torch.no_grad():
            for batch in tqdm(self.train_iter):
                data, targets, pad_lengths = batch[0], batch[1], batch[2]
                data, targets, pad_lengths = self._sort_by_lengths(data, targets, pad_lengths)
                output = self.model(data, pad_lengths)
                batch_label_probabilities = self._batch_label_probs(targets=targets, output=output)
                label_probabilities += batch_label_probabilities

        logging.debug('Len label_probabilities: {}'.format(len(label_probabilities)))
        return label_probabilities

    def convert_to_arpa(self):
        logging.debug('Converting to Approximate 3Gram LM')
        label_probabilities = self._get_label_probabilities()
        self.arpa_converter.convert_to_ngram(label_probabilities)
        self.arpa_converter.write_arpa_format(self.args.output_ngram_lm)
