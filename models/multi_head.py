from .modules.multihead import *
from utils.model_helper import *
import time
from utils.prepare_data import *
from .base_model import BaseModel
import tensorflow as tf

DEFAULT_CONFIG = {
        "max_len": 32,
        "hidden_size": 64,
        "embedding_size": 128,
    }

class MultiheadAttention(BaseModel):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__(config)

        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate

        self.build_model()
        self.init_saver()

    

    def build_model(self):
        self.x = tf.compat.v1.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        embeddings_var = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(params=embeddings_var, ids=self.x)
        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.compat.v1.layers.dense(outputs, units=self.n_class)

        self.loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y))
        self.probabilities = tf.nn.softmax(logits)
        self.prediction = tf.argmax(input=self.probabilities, axis=1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(ys=loss_to_minimize, xs=tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step_tensor,
                                                       name='train_step')
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self.prediction, tf.argmax(input=self.y, axis=1)), tf.float32))

    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)

""" class AttentionClassifier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.compat.v1.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.compat.v1.placeholder(tf.int32, [None])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

    def build_graph(self):
        print("building graph...")
        embeddings_var = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(params=embeddings_var, ids=self.x)
        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.compat.v1.layers.dense(outputs, units=self.n_class)

        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.prediction = tf.argmax(input=tf.nn.softmax(logits), axis=1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(ys=loss_to_minimize, xs=tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")




 """