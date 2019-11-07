from tensorflow.compat.v1.nn.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
from .base_model import BaseModel
from utils.prepare_data import *
from utils.model_helper import *
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

DEFAULT_CONFIG = {
        "max_len": 32,
        "hidden_size": 64,
        "embedding_size": 128,
        "epsilon": 5,
    }

def scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(input_tensor=tf.abs(x), axis=(1, 2), keepdims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(input_tensor=tf.pow(x / alpha, 2), axis=(1, 2), keepdims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def normalize(emb, weights):
    # weights = vocab_freqs / tf.reduce_sum(vocab_freqs) ?? 这个实现没问题吗
    print("Weights: ", weights)
    mean = tf.reduce_sum(input_tensor=weights * emb, axis=0, keepdims=True)
    var = tf.reduce_sum(input_tensor=weights * tf.pow(emb - mean, 2.), axis=0, keepdims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev

class AdversarialClassifier(BaseModel):
    def __init__(self, config):
        super(AdversarialClassifier, self).__init__(config)

        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.epsilon = config.epsilon
        self.vocab_freq = config.vocab_freq
        self.word2idx = config.word2idx

        self.build_model()
        self.init_saver()

    def _add_perturbation(self, embedded, loss):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(
            ys=loss,
            xs=embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = scale_l2(grad, self.epsilon)
        return embedded + perturb

    def _get_freq(self, vocab_freq, word2idx):
        """get a frequency dict format as {word_idx: word_freq}"""
        words = vocab_freq.keys()
        freq = [0] * self.vocab_size
        for word in words:
            word_idx = word2idx.get(word)
            word_freq = vocab_freq[word]
            freq[word_idx] = word_freq
        return freq

    def build_model(self):
        # placeholder
        self.x = tf.compat.v1.placeholder(tf.int32, [None, self.max_len])
        self.y = tf.compat.v1.placeholder(tf.int32, [None, self.n_class])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)
        vocab_freqs = tf.constant(self._get_freq(self.vocab_freq, self.word2idx),
                                  dtype=tf.float32, shape=(self.vocab_size, 1))
        weights = vocab_freqs / tf.reduce_sum(input_tensor=vocab_freqs)
        embeddings_var = tf.Variable(tf.compat.v1.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True, name="embedding_var")
        embedding_norm = normalize(embeddings_var, weights)
        batch_embedded = tf.nn.embedding_lookup(params=embedding_norm, ids=self.x)

        W = tf.Variable(tf.random.normal([self.hidden_size], stddev=0.1))
        W_fc = tf.Variable(tf.random.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0., shape=[self.n_class]))

        def cal_loss_logit(embedded, keep_prob, reuse=True, scope="loss"):
            with tf.compat.v1.variable_scope(scope, reuse=reuse) as scope:
                rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                        BasicLSTMCell(self.hidden_size),
                                        inputs=embedded, dtype=tf.float32)

                # Attention
                H = tf.add(rnn_outputs[0], rnn_outputs[1])  # fw + bw
                M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
                # alpha (bs * sl, 1)
                alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                tf.reshape(W, [-1, 1])))
                r = tf.matmul(tf.transpose(a=H, perm=[0, 2, 1]), tf.reshape(alpha, [-1, self.max_len,
                                                                             1]))  # supposed to be (batch_size * HIDDEN_SIZE, 1)
                r = tf.squeeze(r, [2])
                h_star = tf.tanh(r)
                drop = tf.nn.dropout(h_star, 1 - keep_prob)

                # Fully connected layer（dense layer)
                y_hat = tf.compat.v1.nn.xw_plus_b(drop, W_fc, b_fc)

            return y_hat, tf.reduce_mean(input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=self.y))

        self.logits, self.cls_loss = cal_loss_logit(batch_embedded, self.keep_prob, reuse=False)
        embedding_perturbated = self._add_perturbation(batch_embedded, self.cls_loss)
        adv_logits, self.adv_loss = cal_loss_logit(embedding_perturbated, self.keep_prob, reuse=True)
        self.loss = self.cls_loss + self.adv_loss

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(ys=loss_to_minimize, xs=tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step_tensor,
                                                       name='train_step')
        self.probabilities = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(input=self.probabilities, axis=1)
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self.prediction, tf.argmax(input=self.y, axis=1)), tf.float32))


    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)