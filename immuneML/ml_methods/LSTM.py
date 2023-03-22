import csv
import json
import os
import sys

import datetime
import random
import time
import pyprind

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder


class LSTM(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}
        super(LSTM, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

        self.checkpoint_dir = ""
        self.model_params = {}
        self.historydf = None
        self.initializer = None
        self.generated_sequences = []

    def _get_ml_model(self, batch_size, vocab_size=20, rnn_units=128):

        #This model is the same as Mat's, but the Embedding layer is replaced with a simple
        #Input layer. The input shape and output shape are the same

        x = tf.keras.layers.Input((None, vocab_size), batch_size=batch_size)
        lstm = tf.keras.layers.LSTM(rnn_units,
                                            return_sequences=True,
                                            stateful=True,
                                            recurrent_initializer='glorot_uniform')(x)
        dense2 = tf.keras.layers.Dense(vocab_size)(lstm)
        self.model = tf.keras.models.Model(x, dense2, name="model")

    def split_input_target(self, seq):
        '''
        split input output, return input-output pairs
        :param seq:
        :return:
        '''
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return input_seq, target_seq

    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def _fit(self, dataset, cores_for_training: int = 1, result_path: Path = None):

        """
        Hard values made from previous testing with LSTM. These values may not be suitable for protein sequence
        generation
        """

        PathBuilder.build(result_path)


        params = self._parameters
        rnn_units = 128 if "rnn_units" not in params else params["rnn_units"]
        epochs = 10 if "epochs" not in params else params["epochs"]



        """
        This snippit adds another dimension to the one_hot
        The existing one_hot implementation ignores spaces, and simply returns each sequence
        LSTM requires a data input of a bunch of sequences.
        Here we add a 0 at the start of each one_hot value, and then insert a one_hot array representing
        a space in intervals of "sequence length".
        
        In order to solve the mismatching lenght of sequences, we also remove every instance
        of an array with no 1's.
        """
        dataset = np.insert(dataset, 0, 0, axis=2)
        zero = np.zeros(dataset.shape[2])
        zero[0] = 1
        dataset = np.insert(dataset, dataset.shape[1], zero, axis=1)
        dataset = np.reshape(dataset, (dataset.shape[0] * dataset.shape[1], dataset.shape[2]))
        dataset = np.delete(dataset, dataset.shape[0]-1, axis=0)
        dataset = dataset[np.any(dataset == 1, axis=1)]
        batch_size = 32

        """ LSTM requires lengths of sentences to process, hence, the data is split into chunks of
        an arbitrary length.
        We then map a function that splits the data for x and y input for the fitting.
        The mapping ensures that the input of a chunk is compared to the following chunk.
        This was the same as mat did.
        """
        batches = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(dataset))
        sentence_length = 42

        batches = batches.batch(sentence_length, drop_remainder=True)
        batches = batches.map(self.split_input_target)
        batches = batches.batch(batch_size=batch_size, drop_remainder=True)
        dataset_size = 0
        for _ in batches:
            dataset_size += 1
        # split train, val, test
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        train_dataset = batches.take(train_size)
        test_dataset = batches.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        self.alphabet.insert(0, ' ')
        self.char2idx = {a: i for i, a in enumerate(self.alphabet)}

        train_len = int(dataset.shape[0] * (2 / 3))
        x_train = dataset[:train_len]
        x_test = dataset[train_len:]
        self.initializer = x_train[0]

        vocab_size = len(self.alphabet)

        self._get_ml_model(batch_size, vocab_size, rnn_units)

        """
        Mat applies sparse categorical crossentropy, but this is not possible with one_hot
        """
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")

        self.checkpoint_dir = result_path / f"{self._get_model_filename()}_checkpoints"
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt_{epoch}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            verbose=1,
            save_best_only=False
        )

        history = self.model.fit(train_dataset,
                                 epochs=epochs,
                                 callbacks=[checkpoint_callback],
                                 validation_data=test_dataset,
                                 workers=cores_for_training)

        history_contents = []
        for metric in history.history:
            metric_contents = []
            for i, val in enumerate(history.history[metric]):
                history_content = [val]
                metric_contents.append(history_content)
            history_contents.append([metric, metric_contents])
        self.historydf = pd.DataFrame(history_contents, columns=['metric', 'data'])

        self.model_params = {
            'epochs': epochs,
            'vocab_size': vocab_size,
            'rnn_units': rnn_units
        }

        return self.model

    def generate(self, amount=10, path_to_model: Path = None):

        self._get_ml_model(batch_size=1,
                           vocab_size=self.model_params['vocab_size'],
                           rnn_units=self.model_params['rnn_units'])

        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))

        init = [3]
        hot = tf.one_hot(init, 21)
        hot = tf.expand_dims(hot, 0)

        #Mat resets states, but this causes the values to become "nan"
        #self.model.reset_states()
        for i in range(amount):
            sentence = ""
            while self.alphabet[np.argmax(hot)] != " ":
                hot = self.model(hot)
                sentence = sentence + self.alphabet[np.argmax(hot)]
                if len(sentence) > 42:
                    break
            print(sentence)
            self.generated_sequences.append(sentence)

        return self.generated_sequences
    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        self.model_params = json.load(open(path / f"{self._get_model_filename()}_model_params.json"))
        self.initializer = np.loadtxt(path / f"{self._get_model_filename()}_init.csv")
        self.checkpoint_dir = path / f"{self._get_model_filename()}_checkpoints"

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        print(f'{datetime.datetime.now()}: Writing to file...')
        params_path = path / f"{self._get_model_filename()}_model_params.json"
        init_path = path / f"{self._get_model_filename()}_init.csv"

        model_params_outname = params_path
        np.savetxt(init_path, self.initializer, delimiter=",")
        json.dump(self.model_params, open(model_params_outname, 'w'))

    @staticmethod
    def get_documentation():
        doc = str(LSTM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation(
                "LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc
