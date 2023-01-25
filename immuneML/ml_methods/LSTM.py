import csv
import json
import os

import datetime
import random

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
        self.char2idx = {}
        self.idx2char = {}
        self.max_length = 40
        self.historydf = None
        self.generated_sequences = []

    def split_input_target(self, seq):
        '''
        split input output, return input-output pairs
        :param seq:
        :return:
        '''
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return input_seq, target_seq

    def _get_ml_model(self, cores_for_training: int = 2, X=None, vocab_size=21, rnn_units=128, embedding_dim=256, batch_size=64):

        #put one hot
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])


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

        params = self._parameters

        #default values of optional params
        rnn_units = 128 if "rnn_units" not in params else params["rnn_units"]
        epochs = 10 if "epochs" not in params else params["epochs"]

        vocab_size = len(self._alphabet)
        embedding_dim = 32
        self.max_length = 42
        batch_size = 128
        buffer_size = 1000


        self.char2idx = {u: i for i, u in enumerate(self._alphabet)}
        self.idx2char = np.array(self._alphabet)

        sentences = []
        next_chars = []
        # for i in range(0, len(dataset) - self.max_length, self.max_length):
        #     sentences.append(dataset[i: i + self.max_length])
        #     next_chars.append(dataset[i + self.max_length])
        # print("Number of sequences:", len(sentences))

        # x = np.zeros((len(sentences), self.max_length, len(self._alphabet)), dtype=np.bool)
        # y = np.zeros((len(sentences), len(self._alphabet)), dtype=np.bool)
        # for i, sentence in enumerate(sentences):
        #     for t, char in enumerate(sentence):
        #         x[i, t, self.char2idx[char]] = 1
        #     y[i, self.char2idx[next_chars[i]]] = 1
        tensor_x = tf.data.Dataset.from_tensor_slices([self.char2idx[i] for i in dataset])
        sequences_as_one_hot = tensor_x.batch(self.max_length, drop_remainder=True)
        final_data = sequences_as_one_hot.map(self.split_input_target)
        final_data = final_data.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        dataset_size = 0
        for _ in dataset:
            dataset_size += 1
        print(dataset_size)
        scaller = 1
        # split train, val, test
        train_size = int(0.7 / scaller * dataset_size)
        val_size = int(0.15 / scaller * dataset_size)
        test_size = int(0.15 / scaller * dataset_size)
        print('Trains batches {}, val batches {}, test batches {}'.format(train_size, val_size, test_size))
        train_dataset = final_data.take(train_size)
        test_dataset = final_data.skip(train_size)
        val_dataset = test_dataset.skip(val_size)

        self._get_ml_model(vocab_size=vocab_size, rnn_units=rnn_units, embedding_dim=embedding_dim, batch_size=batch_size)

        for input_example_batch, output_example_batch in final_data.take(1):
            example_batch_predictions = self.model(input_example_batch)
            print(example_batch_predictions.shape)
            sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
            print(sampled_indices)
            sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
            print(''.join(self.idx2char[sampled_indices]))
            print(''.join(self.idx2char[input_example_batch[0].numpy()]))
        ####

        example_loss = self.loss(output_example_batch, example_batch_predictions)
        print(example_loss.numpy().mean())

        self.model.summary()
        self.model.compile(loss=self.loss, optimizer='adam')
        #Prep checkpoint paths
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
                                 validation_data=val_dataset)
        history_contents = []
        for metric in history.history:
            metric_contents = []
            for i, val in enumerate(history.history[metric]):
                history_content = [val]
                metric_contents.append(history_content)
            history_contents.append([metric, metric_contents])
        self.historydf = pd.DataFrame(history_contents, columns=['metric', 'data'])

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
            'epochs': epochs,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'rnn_units': rnn_units
        }

        return self.model

    def generate(self, amount=10, path_to_model: Path = None):

        self._get_ml_model(vocab_size=self.model_params['vocab_size'],
                           rnn_units=self.model_params['rnn_units'],
                           embedding_dim=self.model_params['embedding_dim'],
                           batch_size=1)

        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))

        sentence = "VICTR" #Random sequence chosen from dataset
        print('...Generating with seed: "' + sentence + '"')

        input_vect = [self.char2idx[s] for s in sentence]
        input_vect = tf.expand_dims(input_vect, 0)
        generated_seq = ''
        temperature = 1.
        self.model.reset_states()
        count = 0

        while True:
            prediction = self.model(input_vect)
            prediction = tf.squeeze(prediction, 0)
            prediction = prediction / temperature
            predicted_char = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()
            input_vect = tf.expand_dims([predicted_char], 0)
            if self.idx2char[predicted_char] == ' ':
                count += 1
            if count == amount:
                break
            generated_seq += self.idx2char[predicted_char]


        self.generated_sequences = np.array(generated_seq.split(' '))
        return self.generated_sequences

    def get_params(self):
        return self._parameters

    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        self.model_params = json.load(open(path / f"{self._get_model_filename()}_model_params.json"))
        self.char2idx = json.load(open(path / f"{self._get_model_filename()}_char2idx.json"))
        self.idx2char = list(self.char2idx.keys())
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
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc