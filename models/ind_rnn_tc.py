from modules.indRNN import IndRNNCell
from modules.attention import attention
import time
from utils.prepare_data import *

# Hyperparameter
MAX_DOCUMENT_LENGTH = 256
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 64
ATTENTION_SIZE = 64
lr = 1e-3
BATCH_SIZE = 1024
KEEP_PROB = 0.5
LAMBDA = 1e-3
MAX_LABEL = 15
epochs = 10

# load data
x_train, y_train = load_data("../dbpedia_csv/train.csv", sample_ratio=1)
x_test, y_test = load_data("../dbpedia_csv/test.csv", sample_ratio=1)

# data preprocessing
x_train, x_test,  vocab_size = \
    data_preprocessing_v2(x_train, x_test, MAX_DOCUMENT_LENGTH)
print(vocab_size)

# split dataset to test and dev
x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
print("Validation size: ", dev_size)

graph = tf.Graph()
with graph.as_default():

    batch_x = tf.compat.v1.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.compat.v1.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    embeddings_var = tf.Variable(tf.random.uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(params=embeddings_var, ids=batch_x)
    print(batch_embedded.shape)  # (?, 256, 100)

    cell = IndRNNCell(HIDDEN_SIZE)
    rnn_outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, batch_embedded, dtype=tf.float32)

    # Attention
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    drop = tf.nn.dropout(attention_output, 1 - (1 - (keep_prob)))
    shape = drop.get_shape()

    # Fully connected layer（dense layer)
    W = tf.Variable(tf.random.truncated_normal([shape[1].value, MAX_LABEL], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
    y_hat = tf.compat.v1.nn.xw_plus_b(drop, W, b)

    loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(input=tf.nn.softmax(y_hat), axis=1)
    accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(prediction, tf.argmax(input=batch_y, axis=1)), tf.float32))


with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print("Initialized! ")

    print("Start trainning")
    start = time.time()
    for e in range(epochs):

        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        epoch_finish = time.time()
        print("Validation accuracy: ", sess.run([accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        }))
        print("Epoch time: ", time.time() - epoch_start, "s")

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: 1.0}
            acc = sess.run(accuracy, feed_dict=fd)
            test_acc += acc
            cnt += 1        
    
    print("Test accuracy : %f %%" % ( test_acc / cnt * 100))





