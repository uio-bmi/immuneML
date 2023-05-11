import json
import sys

import datetime
import pyprind

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder


class LSTM(GenerativeModel):
    """
    This is an implementation of Long Short-Term Memory as a generative model.
    This ML method applies the Keras LSTM module to develop a neural network for training. The model is based on the
    paper by Akbar et al. (2021) https://www.biorxiv.org/content/10.1101/2021.07.08.451480v1

    For usage instructions, check :py:obj:`~immuneML.ml_methods.GenerativeModel.GenerativeModel`.

    Arguments specific for LSTM:

        batch_size: size of batches trained on at a time
        rnn_units: number of units in the LSTM neural net
        embedding_dim: Dimensionality of the embedding layer in the model, encoding the data
        epochs: number of epochs to train the model
        max_sequence_length: data preprocessing sequence limiter
        buffer_size: data preprocessing buffer


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_lstm:
            LSTM:
                # params
                amount: 100 # defaults to 100
                sequence_type: sequence_aas # specify according to dataset applied
                rnn_units: 512
                epochs: 10
                batch_size: 32
        # alternative way to define ML method with default values:
        my_default_lstm: LSTM

    """

    def get_classes(self) -> list:
        pass

    def __init__(self, **parameters):
        parameters = parameters if parameters is not None else {}
        super(LSTM, self).__init__(parameters=parameters)

        self._checkpoint_dir = ""
        self.model_params = {}

        self.historydf = None
        self.generated_sequences = []
        self.alphabet.append(" ")

        self._max_sequence_length = parameters["max_sequence_length"]
        self._rnn_units = parameters["rnn_units"]
        self._epochs = parameters["epochs"]
        self._embedding_dim = parameters["embedding_dim"]
        self._batch_size = parameters["batch_size"]
        self._buffer_size = parameters["buffer_size"]
        self._first_sequence = None

        self._vocab_size = len(self.alphabet)

    @staticmethod
    def _split_input_target(seq):

        input_seq = seq[:-1]
        target_seq = seq[1:]
        return input_seq, target_seq

    def _get_ml_model(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self._vocab_size,
                                      self._embedding_dim,
                                      batch_input_shape=[self._batch_size, None]),
            tf.keras.layers.LSTM(self._rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self._vocab_size)
        ])

        return model

    @staticmethod
    def loss(labels, logits):

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss

    def _fit(self, dataset, cores_for_training: int = 1):

        self._first_sequence = []
        for i in dataset:
            if i != 20:
                self._first_sequence.append(i)
            else:
                self._first_sequence.append(i)
                break
        tensor_x = tf.data.Dataset.from_tensor_slices(dataset)
        sequence_batches = tensor_x.batch(self._max_sequence_length, drop_remainder=True)
        final_data = sequence_batches.map(self._split_input_target)
        final_data = final_data.shuffle(self._buffer_size).batch(self._batch_size, drop_remainder=True)
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

        history = model.fit(train_dataset,
                                 epochs=self._epochs,
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
            'epochs': self._epochs,
            'batch_size': self._batch_size,
            'buffer_size': self._buffer_size,
            'vocab_size': self._vocab_size,
            'embedding_dim': self._embedding_dim,
            'rnn_units': self._rnn_units,
            'first_sequence': self._first_sequence
        }

        return model

    def generate(self):

        self._batch_size = 1  # When generating, we only generate one batches of size 1
        model = self._get_ml_model()

        model.load_weights(self._checkpoint_dir)

        input_vect = tf.expand_dims(self._first_sequence, 0)
        generated_seq = ''
        model.reset_states()
        count = 0
        bar = pyprind.ProgBar(self._amount, bar_char="=", stream=sys.stdout, width=100)

        while True:
            prediction = model(input_vect)
            prediction = tf.squeeze(prediction, 0)
            predicted_char = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()
            input_vect = tf.expand_dims([predicted_char], 0)
            if self.alphabet[predicted_char] == ' ':
                count += 1
                bar.update()
            if count == self._amount:
                break
            generated_seq += self.alphabet[predicted_char]

        generated_sequences = np.array(generated_seq.split(' '))
        return generated_sequences

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        from immuneML.encodings.char_to_int.CharToIntEncoder import CharToIntEncoder

        return [CharToIntEncoder]

    def load(self, path: Path, details_path: Path = None):

        self.model_params = json.load(open(path / f"{self._get_model_filename()}_model_params.json"))
        self._rnn_units = self.model_params["rnn_units"]
        self._embedding_dim = self.model_params["embedding_dim"]
        self._first_sequence = self.model_params["first_sequence"]


        self._vocab_size = len(self.alphabet)
        self.char2idx = json.load(open(path / f"{self._get_model_filename()}_char2idx.json"))
        self.alphabet = list(self.char2idx.keys())
        self._checkpoint_dir = path / f"{self._get_model_filename()}_weights"

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        self._checkpoint_dir = path / f"{self._get_model_filename()}_weights"
        self.model.save_weights(self._checkpoint_dir)

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
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.GenerativeModel.GenerativeModel`.": GenerativeModel.get_usage_documentation(
                "LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc


