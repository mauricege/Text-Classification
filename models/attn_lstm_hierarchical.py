from .modules.attention import attention
from tensorflow.compat.v1.nn.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
from utils.prepare_data import *
from utils.model_helper import *
from .base_model import BaseModel
import tensorflow as tf
tf.compat.v1.disable_eager_execution()



DEFAULT_CONFIG = {'embedding_size': 128,
                'hidden_size': 64,
                'attention_size': 64,
                'lmbda': 0.0001}




class AttentionLSTMHierarchical(BaseModel):
    def __init__(self, config):
        super(AttentionLSTMHierarchical, self).__init__(config)

        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.lmbda = config.lmbda
        self.attention_size = config.attention_size

        self.build_model()
        self.init_saver()

    

    def build_model(self):
        # placeholder

        self.x = tf.compat.v1.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        embeddings_var = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), trainable=True)
        batch_embedded = tf.nn.embedding_lookup(params=embeddings_var, ids=self.x)
        # print(batch_embedded.shape)  # (?, 256, 100)
        rnn_outputs, _ = tf.compat.v1.nn.dynamic_rnn(BasicLSTMCell(self.hidden_size), batch_embedded, dtype=tf.float32)

        # Attention
        attention_output, alphas = attention(rnn_outputs, self.attention_size, return_alphas=True)
        drop = tf.nn.dropout(attention_output, 1 - self.keep_prob)
        shape = drop.get_shape()

        # Fully connected layer（dense layer)
        W = tf.Variable(tf.random.truncated_normal([shape[1], self.n_class], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.compat.v1.nn.xw_plus_b(drop, W, b)


        self.loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=self.y))
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step_tensor,
                                                       name='train_step')

        # Accuracy metric
        self.probabilities = tf.nn.softmax(y_hat)
        self.prediction = tf.argmax(input=self.probabilities, axis=1)
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self.prediction, tf.argmax(input=self.y, axis=1)), tf.float32))
        


    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)

