from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.compat.v1.nn.rnn_cell import BasicLSTMCell
from utils.prepare_data import *
from .base_model import BaseModel
from .base_train import BaseTrain
from tqdm import tqdm
import time
from os.path import join
from os import makedirs
from utils.model_helper import *
from utils.logger import Logger
from bunch import Bunch
from sklearn.metrics import precision_recall_fscore_support, classification_report
import tensorflow as tf
from keras_preprocessing.text import tokenizer_from_json
import pickle
import pandas as pd
tf.compat.v1.disable_eager_execution()

DEFAULT_CONFIG = {
        "max_len": 32,
        "hidden_size": 64,
        "embedding_size": 128,
        "n_class": 10,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_epochs": 20,
        "summary_dir": "log",
        "checkpoint_dir": "ckpts/",
        "max_to_keep": 5
    }

class AttentionBiLSTM(BaseModel):
    def __init__(self, config):
        super(AttentionBiLSTM, self).__init__(config)
        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.build_model()
        self.init_saver()

    def build_model(self):
        # placeholder
        self.x = tf.compat.v1.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.compat.v1.placeholder(tf.int32, [None, self.n_class])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)
        # Word embedding
        embeddings_var = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        self.batch_embedded = tf.nn.embedding_lookup(params=embeddings_var, ids=self.x)

        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=self.batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random.normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(a=H, perm=[0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r, [2])
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, 1 - (self.keep_prob))

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.random.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.compat.v1.nn.xw_plus_b(h_drop, FC_W, FC_b)

        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=self.y))

        # prediction
        self.probabilities = tf.nn.softmax(y_hat)
        self.prediction = tf.argmax(input=self.probabilities, axis=1)

        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self.prediction, tf.argmax(input=self.y, axis=1)), tf.float32))

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(ys=loss_to_minimize, xs=tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step_tensor,
                                                      name='train_step')

    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)