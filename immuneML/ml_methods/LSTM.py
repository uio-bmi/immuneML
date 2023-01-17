import csv
import json
import os

import datetime
import random

import numpy as np
import pandas as pd
import tensorflow as tf

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
        self.model_params = {}
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.max_length = 40
        self.rnn_units = 1024

    def _get_ml_model(self, cores_for_training: int = 2, X=None):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_length, self.vocab_size)),
            tf.keras.layers.LSTM(self.rnn_units),
            tf.keras.layers.Dense(self.vocab_size, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    @staticmethod
    def loss(labels, logits):
        '''
        loss function for the net output
        :param labels:
        :param logits:
        :return:
        '''
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss


    def _fit(self, dataset, cores_for_training: int = 1, result_path: Path = None):

        """
        Hard values made from previous testing with LSTM. These values may not be suitable for protein sequence
        generation
        """

        PathBuilder.build(result_path)

        self.vocab_size = len(self._alphabet)
        self.rnn_units = 1024
        self.max_length = 42
        step = 3
        epochs = 20
        batch_size = 64

        self.char2idx = {u: i for i, u in enumerate(self._alphabet)}
        self.idx2char = np.array(self._alphabet)

        sentences = []
        next_chars = []
        for i in range(0, len(dataset) - self.max_length, step):
            sentences.append(dataset[i: i + self.max_length])
            next_chars.append(dataset[i + self.max_length])
        print("Number of sequences:", len(sentences))

        x = np.zeros((len(sentences), self.max_length, len(self._alphabet)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self._alphabet)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char2idx[char]] = 1
            y[i, self.char2idx[next_chars[i]]] = 1

        self._get_ml_model()

        self.model.summary()

        #Prep checkpoint paths
        checkpoints_path = result_path / f"{self._get_model_filename()}_checkpoints.csv"
        checkpoint_prefix = os.path.join(checkpoints_path, 'ckpt_{epoch}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            verbose=1,
            save_best_only=False
        )

        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint_callback])

        # for epoch in range(epochs):
        #     self.model.fit(x, y, batch_size=batch_size, epochs=1, callbacks=[checkpoint_callback])
        #     print()
        #     print("Generating text after epoch: %d" % epoch)
        #
        #     start_index = random.randint(0, len(dataset) - self.max_length - 1)
        #     for diversity in [0.2, 0.5, 1.0, 1.2]:
        #         print("...Diversity:", diversity)
        #
        #         generated = ""
        #         sentence = ''.join(dataset[start_index: start_index + self.max_length])
        #         print('...Generating with seed: "' + sentence + '"')
        #
        #         for i in range(100):
        #             x_pred = np.zeros((1, self.max_length, self.vocab_size))
        #             for t, char in enumerate(sentence):
        #                 x_pred[0, t, self.char2idx[char]] = 1.0
        #             preds = self.model.predict(x_pred, verbose=0)[0]
        #             next_index = self.sample(preds, diversity)
        #             next_char = self.idx2char[next_index]
        #             sentence = sentence[1:] + next_char
        #             generated += next_char
        #
        #         print("...Generated: ", generated)
        #         print()

        self.model_params = {
            'nb_epoch': epochs,
            'batch_size': batch_size,
            'vocab_size': self.vocab_size,
            'rnn_units': self.rnn_units
        }

        return self.model

    def generate(self, amount=10, path_to_model: Path = None):

        generated = ""
        sentence = "NDARTDNAAAYLHWVDFNLQ" #Random sequence chosen from dataset
        print('...Generating with seed: "' + sentence + '"')


        for i in range(7000):  # Make one sequence at a time
            x_pred = np.zeros((1, self.max_length, self.vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char2idx[char]] = 1.0
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, 1)
            next_char = self.idx2char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print(generated)

        return generated

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        model_params = json.load(path / f"{self._get_model_filename()}_model_params.csv")
        self.char2idx = json.load(path / f"{self._get_model_filename()}_char2idx.csv")
        self.idx2char = list(self.char2idx.keys())

        checkpoint_dir = path / f"{self._get_model_filename()}_checkpoints.csv"
        self.vocab_size = model_params['vocab_size']
        self.rnn_units = model_params['rnn_units']
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        print(f'{datetime.datetime.now()}: Writing to file...')
        params_path = path / f"{self._get_model_filename()}_model_params.csv"
        char2idx_path = path / f"{self._get_model_filename()}_char2idx.csv"


        model_params_outname = params_path
        char2idx_outname = char2idx_path
        json.dump(self.model_params, open(model_params_outname, 'w'))
        json.dump(self.char2idx, open(char2idx_outname, 'w'))


    @staticmethod
    def get_documentation():
        doc = str(LSTM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc