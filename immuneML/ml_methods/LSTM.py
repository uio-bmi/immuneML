import csv
import yaml
import datetime

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


    def _get_ml_model(self, cores_for_training: int = 2, X=None):


        """
        :param cores_for_training:
        :param X:
        :return: keras.Sequential object

        The initial parameters set have been determined through testing on a previous project using LSTM. It is
        worthwhile considering changing these.
        """
        embedding_dim = 256
        rnn_units = 1024
        seq_length = 42  # window size (w)
        batch_size = 64
        buffer_size = 1000

        vocab_size = len(self._alphabet)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])

        return self.model

    def loss(self, labels, logits):
        '''
        loss function for the net output
        :param labels:
        :param logits:
        :return:
        '''
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss

    # split input target
    def split_input_target(self, seq):
        '''
        split input output, return input-output pairs
        :param seq:
        :return:
        '''
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return input_seq, target_seq

    def _fit(self, encoded_data, cores_for_training: int = 1):

        nb_epoch = 20

        char_dataset = tf.data.Dataset.from_tensor_slices(encoded_data)

        sequences = char_dataset.batch(self._length_of_sequence + 1, drop_remainder=True)

        dataset = sequences.map(self.split_input_target)
        batch_size = 64
        buffer_size = 1000
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        # get dataset size since there is no dim/shape attributes in tf.dataset
        dataset_size = 0
        for _ in dataset:
            dataset_size += 1
        scaller = 1
        # split train, val, test
        train_size = int(0.7 / scaller * dataset_size)
        val_size = int(0.15 / scaller * dataset_size)
        test_size = int(0.15 / scaller * dataset_size)
        print('Trains batches {}, val batches {}, test batches {}'.format(train_size, val_size, test_size))

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)

        self.model = self._get_ml_model()

        self.model.summary()
        # sanity checks
        for input_example_batch, output_example_batch in dataset.take(1):
            example_batch_predictions = self.model(input_example_batch)
            print(example_batch_predictions.shape)
            sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
            print(sampled_indices)
            sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
        ####

        example_loss = self.loss(output_example_batch, example_batch_predictions)
        print(example_loss.numpy().mean())
        self.model.compile(optimizer='adam', loss=self.loss)

        self.model.fit(train_dataset, epochs=nb_epoch,
                            validation_data=val_dataset)

        return self.model

    def generate(self, amount=10, path_to_model: Path = None):

        test = self.model("VICTR")

        test2 = self.model.predict(self._length_of_sequences)


        print(test)
        print(test2)

        return test2

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        name = f"{self._get_model_filename()}.csv"
        file_path = path / name
        if file_path.is_file():
            dataframe = file_path
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}"
                                    f". Check if the path to the {name} file is properly set.")

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        if params_path.is_file():
            with params_path.open("r") as file:
                desc = yaml.safe_load(file)
                for param in ["feature_names"]:
                    if param in desc:
                        setattr(self, param, desc[param])

    def store(self, path: Path, feature_names=None, details_path: Path = None):

        pass

    @staticmethod
    def get_documentation():
        doc = str(LSTM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc