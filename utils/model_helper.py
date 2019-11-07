import numpy as np


def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.y: batch[1],
                 model.keep_prob: .5}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.y: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'prediction': model.prediction,
        'probabilities': model.probabilities,
        'label': model.y,
        'acc': model.accuracy,
        'global_step': model.global_step_tensor
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    to_return = {
        'prediction': model.prediction,
        'probabilities': model.probabilities,
        'loss': model.loss,
        'acc': model.accuracy
    }
    return sess.run(to_return, feed_dict)


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)
