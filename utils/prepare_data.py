import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer

names = ["class", "title", "content"]


def to_one_hot(y):
    lb = MultiLabelBinarizer()
    y_transformed = lb.fit_transform(np.array(y).reshape(-1, 1))
    return y_transformed


def load_data(file_name, sample_ratio=1, one_hot=True, shuffle=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name)
    if shuffle:
        shuffle_csv = csv_file.sample(frac=sample_ratio)
    else:
        shuffle_csv = csv_file
    titles = pd.Series(shuffle_csv["title"])
    x = pd.Series(shuffle_csv["content"])
    if 'class' in shuffle_csv.columns:
        y = pd.Series(shuffle_csv["class"])
    else:
        y = None
        return titles, x, y
    if one_hot:
        y = to_one_hot(y)
    return titles, x, y


def data_preprocessing(data, max_len, max_words=50000, tokenizer=None):
    if not tokenizer:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
    idx = tokenizer.texts_to_sequences(data)
    #test_idx = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(idx, maxlen=max_len, padding='post', truncating='post')
    #test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return padded, max_words + 2, tokenizer


def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size, shuffle_data=True):
    """Generator to yield batches"""
    # Shuffle data first.
    if shuffle_data:
        shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    else:
        shuffled_X, shuffled_Y = data_X, data_Y
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
