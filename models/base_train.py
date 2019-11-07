import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.prepare_data import *
from utils.model_helper import *
from sklearn.metrics import precision_recall_fscore_support, classification_report



class BaseTrain:
    def __init__(self, sess, model, train_data, config, logger, val_data=None, restore=False, no_labels=False):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.no_labels = no_labels
        self.train_data = train_data
        self.val_data = val_data
        self.init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        self.sess.run(self.init)
        if restore:
            self.model.load(sess)

        self.feed_dict_train = None
        self.feed_dict_val = None
        
        if self.train_data is not None:
            self.feed_dict_train = fill_feed_dict(self.train_data[0], self.train_data[1], self.config.batch_size)
        if val_data is not None:
            self.feed_dict_val = fill_feed_dict(self.val_data[0], self.val_data[1], self.config.batch_size, shuffle_data=False)
        self.best_result = 0

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            print(f'\nEpoch {cur_epoch + 1}')
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def eval(self):
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        print(f'\nValidation after epoch {cur_epoch + 1}:')
        return self.eval_epoch()

    def train_epoch(self):
        loop = tqdm(range(self.train_data[0].shape[0] // self.config.batch_size))
        losses = []
        accs = []
        for _ in loop:
            cur_loss, cur_acc = self.train_step()
            losses.append(cur_loss)
            accs.append(cur_acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print(f'Training: \tloss: {loss} \taccuracy: {acc}')

        
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict_train = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict_train)
        self.feed_dict_train = fill_feed_dict(self.train_data[0], self.train_data[1], self.config.batch_size)

        if self.val_data is not None:
            summaries_dict_val, _, _ = self.eval_epoch()
            self.logger.summarize(cur_it, summaries_dict=summaries_dict_val, summarizer='val')
            if summaries_dict_val['recall_macro'] > self.best_result:
                print(f'Macro Recall increased from {self.best_result} to {summaries_dict_val["recall_macro"]}.')
                self.model.save(self.sess)
                self.best_result = summaries_dict_val['recall_macro']
            else:
                print(f'Macro Recall did not increase from {self.best_result}.')

    def eval_epoch(self):
        assert self.val_data is not None, 'No validation data.'
        losses = []
        accs = []
        predictions = []
        probabilities = []
        trues = []
        loop = tqdm(range(self.val_data[0].shape[0] // self.config.batch_size))
        for _ in loop:
            cur_loss, cur_acc, cur_predictions, cur_probabilities, true = self.eval_step()
            losses.append(cur_loss)
            accs.append(cur_acc)
            trues.append(np.array(true))
            predictions.append(np.array(cur_predictions))
            probabilities.append(np.array(cur_probabilities))
        loss = np.mean(losses)
        acc = np.mean(accs)
        predictions = np.concatenate(predictions)
        probabilities = np.concatenate(probabilities)
        if not self.no_labels:
            trues = np.concatenate(trues)
            y_true = np.argmax(trues, axis=1)
            precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_true=y_true, y_pred=predictions, average='macro')
            precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(y_true=y_true, y_pred=predictions, average='micro')
            print(f'Validation: \tloss: {loss} \taccuracy: {acc}')
            print(classification_report(y_true=y_true, y_pred=predictions))
            summaries_dict_val = {
            'loss': loss,
            'acc': acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'fscore_macro': fscore_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'fscore_micro': fscore_micro
            }
        else:
            summaries_dict_val = None
        self.feed_dict_val = fill_feed_dict(self.val_data[0], self.val_data[1], self.config.batch_size, shuffle_data=False)
        return summaries_dict_val, predictions, probabilities

    def train_step(self):
        batch_x, batch_y = next(self.feed_dict_train)
        return_dict = run_train_step(self.model, self.sess, (batch_x, batch_y))
        return return_dict['loss'], return_dict['acc']

    def eval_step(self):
        batch_x, batch_y = next(self.feed_dict_val)
        return_dict = run_eval_step(self.model, self.sess, (batch_x, batch_y))
        return return_dict['loss'], return_dict['acc'], return_dict['prediction'], return_dict['probabilities'], batch_y