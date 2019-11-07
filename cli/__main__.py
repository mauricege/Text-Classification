import click
import pkg_resources
import warnings
import shutil
import json
import models.attn_bi_lstm
import models.adversarial_abblstm
import models.attn_lstm_hierarchical
import models.cnn
import time
import pickle
import tensorflow as tf
import pandas as pd
from utils.prepare_data import *
from models.base_model import BaseModel
from models.base_train import BaseTrain
from tqdm import tqdm
from os.path import join, exists, basename, splitext
from os import makedirs, listdir
from utils.model_helper import *
from utils.logger import Logger
from bunch import Bunch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from keras_preprocessing.text import tokenizer_from_json
from json import JSONEncoder

VERSION = "0.0.1"


def clear_model_dir(dir):
    shutil.rmtree(dir)
    makedirs(dir)


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, 'to_json'):
            return obj_to_encode.to_json()

        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()

        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.to_list()

        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)


@click.group()
@click.option('-v', '--verbose', count=True)
@click.version_option(VERSION)
def cli(verbose):
    click.echo('Verbosity: %s' % verbose)


@click.group()
@click.pass_context
def attentionBiLSTM(ctx):
    ctx.obj['model'] = models.attn_bi_lstm.AttentionBiLSTM
    ctx.obj['config'] = models.attn_bi_lstm.DEFAULT_CONFIG

@click.group()
@click.pass_context
def adversarialClassifier(ctx):
    ctx.obj['model'] = models.adversarial_abblstm.AdversarialClassifier
    ctx.obj['config'] = models.adversarial_abblstm.DEFAULT_CONFIG

@click.group()
@click.pass_context
def attentionLSTMHierarchical(ctx):
    ctx.obj['model'] = models.attn_lstm_hierarchical.AttentionLSTMHierarchical
    ctx.obj['config'] = models.attn_lstm_hierarchical.DEFAULT_CONFIG

@click.group()
@click.pass_context
def cnn(ctx):
    ctx.obj['model'] = models.cnn.CNN
    ctx.obj['config'] = models.cnn.DEFAULT_CONFIG

@click.command(
    help=
    'Train a text classification model. The trained model is saved and can be used for evaluation.'
)
@click.argument('training-data',
                nargs=1,
                type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-md',
              '--model-dir',
              required=True,
              help='Directory for saving and restoring model and results.',
              type=click.Path(file_okay=False, writable=True))
@click.option(
    '-vd',
    '--validation-data',
    help=
    'Define a separate validation partition that should be used for model optimization.',
    type=click.Path(dir_okay=False, readable=True))
@click.option('-y', '--yes', help='Accept all prompts.', is_flag=True)
@click.option('-bs',
              '--batch-size',
              type=int,
              default=32,
              help='Batch size for training.')
@click.option('-e',
              '--epochs',
              type=int,
              default=20,
              help='Number of training epochs.')
@click.option('-lr',
              '--learning-rate',
              type=float,
              default=1e-3,
              help='Learning rate for optimisation.')
@click.option('-mtk',
              '--max-to-keep',
              type=int,
              default=5,
              help='Maximum number of checkpoints to keep during training.')
@click.option('-ml',
              '--max-len',
              type=int,
              default=32,
              help='Maximum sequence length.')
@click.pass_context
def train(ctx,
          training_data,
          model_dir,
          validation_data,
          yes,
          batch_size=32,
          epochs=20,
          learning_rate=1e-3,
          max_to_keep=5,
          max_len=32):
    if exists(model_dir) and listdir(model_dir) is not None:
        if not yes:
            click.confirm(
                f'{model_dir} exists and is not empty. Running a new experiment here deletes the existing one. Continue anyway?',
                abort=True)
        clear_model_dir(model_dir)
    makedirs(model_dir, exist_ok=True)
    MODEL = ctx.obj['model']
    CONFIG = ctx.obj['config']
    # load data
    _, x_train, y_train = load_data(training_data, sample_ratio=0.01)
    x_train, vocab_size, tokenizer = \
        data_preprocessing(x_train, max_len=max_len, tokenizer=None)
    print("train size: ", len(x_train))
    print("vocab size: ", vocab_size)

    if validation_data is not None:
        _, x_dev, y_dev = load_data(validation_data, shuffle=False)
        x_dev, vocab_size, tokenizer = \
        data_preprocessing(x_dev, max_len=max_len, tokenizer=tokenizer)
    else:
        x_train, x_dev, y_train, y_dev, dev_size, _ = \
            split_dataset(x_train, y_train, 0.1)
        print("Validation Size: ", dev_size)

    CONFIG["vocab_size"] = vocab_size
    CONFIG["n_class"] = y_train.shape[1]
    CONFIG["batch_size"] = batch_size
    CONFIG["learning_rate"] = learning_rate
    CONFIG["num_epochs"] = epochs
    CONFIG["summary_dir"] = join(model_dir, "log")
    CONFIG["checkpoint_dir"] = join(model_dir, "ckpts/")
    CONFIG["max_to_keep"] = max_to_keep
    CONFIG["max_len"] = max_len
    CONFIG["vocab_freq"] = tokenizer.word_docs
    CONFIG["word2idx"] = tokenizer.word_index



    # save config and tokenizer
    with open(join(model_dir, 'tokenizer.json'), 'w') as f:
        f.write(tokenizer.to_json())
    with open(join(model_dir, 'model.config'), 'wb') as f:
        pickle.dump(CONFIG, f)
    CONFIG = Bunch(CONFIG)
    sess = tf.compat.v1.Session()
    model = MODEL(CONFIG)
    logger = Logger(sess, CONFIG)
    trainer = BaseTrain(sess,
                        model, (x_train, y_train),
                        CONFIG,
                        logger,
                        val_data=(x_dev, y_dev))
    trainer.train()


@click.command(help='Make predictions with a saved model.')
@click.argument('prediction-data',
                nargs=1,
                type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-md',
              '--model-dir',
              required=True,
              help='Directory of the trained model.',
              type=click.Path(file_okay=False, writable=True))
@click.pass_context
def predict(ctx, prediction_data, model_dir):
    MODEL = ctx.obj['model']
    titles_test, x_test, y_test = load_data(prediction_data, shuffle=False)
    labels_available = True
    with open(join(model_dir, 'tokenizer.json')) as f:
        json_string = f.read()
        tokenizer = tokenizer_from_json(json_string)
    with open(join(model_dir, 'model.config'), 'rb') as f:
        config = pickle.load(f)
    makedirs(model_dir, exist_ok=True)
    x_test, _, tokenizer = \
        data_preprocessing(x_test, max_len=config['max_len'], tokenizer=tokenizer)
    config['batch_size'] = 1
    config = Bunch(config)
    sess = tf.compat.v1.Session()
    model = MODEL(config)
    if y_test is None:
        y_test = np.zeros((x_test.shape[0], model.y.shape[1]))
        labels_available = False
    logger = Logger(sess, config)
    trainer = BaseTrain(sess,
                        model,
                        None,
                        config,
                        logger,
                        val_data=(x_test, y_test),
                        restore=True,
                        no_labels=True)

    _, predictions, probabilities = trainer.eval()
    frame_dict = {'title': titles_test, 'prediction': predictions}
    for i in range(probabilities.shape[1]):
        frame_dict[f'probability_{i}'] = probabilities[:, i]
    if labels_available:
        frame_dict[f'true'] = np.argmax(y_test, axis=1)
    df = pd.DataFrame.from_dict(frame_dict)
    print(df)
    df.to_csv(join(model_dir,
                   f'predictions-for-"{basename(prediction_data)}".csv'),
              index=False)


@click.command(help='Evaluate a saved model.')
@click.argument('evaluation-data',
                nargs=1,
                type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('-md',
              '--model-dir',
              required=True,
              help='Directory of the trained model.',
              type=click.Path(file_okay=False, writable=True))
@click.option('-bs',
              '--batch-size',
              type=int,
              default=32,
              help='Batch size for training.')
@click.pass_context
def eval(ctx, evaluation_data, model_dir, batch_size):
    MODEL = ctx.obj['model']
    _, x_test, y_test = load_data(evaluation_data,
                                  sample_ratio=1,
                                  shuffle=False)
    with open(join(model_dir, 'tokenizer.json')) as f:
        json_string = f.read()
        tokenizer = tokenizer_from_json(json_string)
    with open(join(model_dir, 'model.config'), 'rb') as f:
        config = pickle.load(f)
    makedirs(model_dir, exist_ok=True)
    x_test, _, tokenizer = \
        data_preprocessing(x_test, max_len=config['max_len'], tokenizer=tokenizer)

    config['batch_size'] = batch_size

    config = Bunch(config)
    sess = tf.compat.v1.Session()
    model = MODEL(config)
    logger = Logger(sess, config)
    trainer = BaseTrain(sess,
                        model,
                        None,
                        config,
                        logger,
                        val_data=(x_test, y_test),
                        restore=True)

    summaries_dict, _, _ = trainer.eval()
    print(summaries_dict)
    with open(
            join(model_dir,
                 f'evaluation-on-"{basename(evaluation_data)}".json'),
            'w') as fp:
        json.dump(summaries_dict, fp, cls=CustomJSONEncoder)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    attentionBiLSTM.add_command(train)
    attentionBiLSTM.add_command(predict)
    attentionBiLSTM.add_command(eval)
    cli.add_command(attentionBiLSTM)

    adversarialClassifier.add_command(train)
    adversarialClassifier.add_command(predict)
    adversarialClassifier.add_command(eval)
    cli.add_command(adversarialClassifier)

    attentionLSTMHierarchical.add_command(train)
    attentionLSTMHierarchical.add_command(predict)
    attentionLSTMHierarchical.add_command(eval)
    cli.add_command(attentionLSTMHierarchical)

    cnn.add_command(train)
    cnn.add_command(predict)
    cnn.add_command(eval)
    cli.add_command(cnn)

    cli(obj={})
