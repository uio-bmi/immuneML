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

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder


class LSTM(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(LSTM, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

        self.checkpoint_dir = ""
        self.model_params = {}
        self.max_sequence_length = parameters["max_sequence_length"]
        self.historydf = None
        self.generated_sequences = []
        self.alphabet.append(" ")
        self.char2idx = {u: i for i, u in enumerate(self.alphabet)}

        self.rnn_units = self._parameters["rnn_units"]
        self.epochs = self._parameters["epochs"]
        self.embedding_dim = self._parameters["embedding_dim"]
        self.batch_size = self._parameters["batch_size"]
        self.buffer_size = self._parameters["buffer_size"]
        self.first_sequence = None

        self.vocab_size = len(self.alphabet)

    def split_input_target(self, seq):
        '''
        split input output, return input-output pairs
        :param seq:
        :return:
        '''
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return input_seq, target_seq

    def _get_ml_model(self):

        # put one hot
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      batch_input_shape=[self.batch_size, None]),
            tf.keras.layers.LSTM(self.rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self.vocab_size)
        ])

        return model

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

        PathBuilder.build(result_path)
        self.first_sequence = []
        for i in dataset:
            if i != 20:
                self.first_sequence.append(i)
            else:
                self.first_sequence.append(i)
                break
        tensor_x = tf.data.Dataset.from_tensor_slices(dataset)
        sequence_batches = tensor_x.batch(self.max_sequence_length, drop_remainder=True)
        final_data = sequence_batches.map(self.split_input_target)
        final_data = final_data.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
        dataset_size = 0
        for _ in dataset:
            dataset_size += 1
        # split train, val, test
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        train_dataset = final_data.take(train_size)
        test_dataset = final_data.skip(train_size)
        val_dataset = test_dataset.skip(val_size)

        model = self._get_ml_model()

        model.summary()
        model.compile(loss=self.loss, optimizer='adam')
        # Prep checkpoint paths
        self.checkpoint_dir = result_path / f"{self._get_model_filename()}_checkpoints"
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt_{epoch}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            verbose=1,
            save_best_only=False
        )

        history = model.fit(train_dataset,
                                 epochs=self.epochs,
                                 callbacks=[checkpoint_callback],
                                 validation_data=val_dataset)

        history_contents = []
        for metric in history.history:
            metric_contents = []
            for i, val in enumerate(history.history[metric]):
                history_content = [val]
                metric_contents.append(history_content)
            history_contents.append([metric, metric_contents])
        self.historydf = pd.DataFrame(history_contents, columns=['metric', 'data'])

        self.model_params = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'rnn_units': self.rnn_units,
            'first_sequence': self.first_sequence
        }

        return model

    def generate(self, amount=10, path_to_model: Path = None):

        self.batch_size = 1  # When generating, we only generate one batches of size 1
        model = self._get_ml_model()

        model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))

        input_vect = tf.expand_dims(self.first_sequence, 0)
        generated_seq = ''
        model.reset_states()
        count = 0
        bar = pyprind.ProgBar(amount, bar_char="=", stream=sys.stdout, width=100)

        while True:
            prediction = model(input_vect)
            prediction = tf.squeeze(prediction, 0)
            predicted_char = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()
            input_vect = tf.expand_dims([predicted_char], 0)
            if self.alphabet[predicted_char] == ' ':
                count += 1
                bar.update()
            if count == amount:
                break
            generated_seq += self.alphabet[predicted_char]

        generated_sequences = np.array(generated_seq.split(' '))
        return generated_sequences

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        self.model_params = json.load(open(path / f"{self._get_model_filename()}_model_params.json"))
        self.rnn_units = self.model_params["rnn_units"]
        self.embedding_dim = self.model_params["embedding_dim"]
        self.first_sequence = self.model_params["first_sequence"]


        self.vocab_size = len(self.alphabet)
        self.char2idx = json.load(open(path / f"{self._get_model_filename()}_char2idx.json"))
        self.alphabet = list(self.char2idx.keys())
        self.checkpoint_dir = path / f"{self._get_model_filename()}_checkpoints"

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        print(f'{datetime.datetime.now()}: Writing to file...')
        params_path = path / f"{self._get_model_filename()}_model_params.json"
        char2idx_path = path / f"{self._get_model_filename()}_char2idx.json"

        model_params_outname = params_path
        char2idx_outname = char2idx_path
        json.dump(self.model_params, open(model_params_outname, 'w'))
        json.dump(self.char2idx, open(char2idx_outname, 'w'))

    @staticmethod
    def get_documentation():
        doc = str(LSTM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation(
                "LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc


