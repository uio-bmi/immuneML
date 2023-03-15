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

class VAE(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}
        super(VAE, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

        self.encoder = None
        self.decoder = None
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

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0,
                                                 stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
        return random_sample

    def _get_ml_model(self, seq_length, vocab_size, latent_space_dim=2):

        # Encoder
        x = tf.keras.layers.Input(shape=(seq_length, vocab_size), name="encoder_input")

        encoder_conv_layer1 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding="same", strides=1,
                                                             name="encoder_conv_1")(x)
        encoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
        encoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)

        encoder_conv_layer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", strides=1,
                                                             name="encoder_conv_2")(encoder_activ_layer1)
        encoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
        encoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_2")(encoder_norm_layer2)

        encoder_conv_layer3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", strides=2,
                                                             name="encoder_conv_3")(encoder_activ_layer2)
        encoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
        encoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_3")(encoder_norm_layer3)

        encoder_conv_layer4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", strides=2,
                                                             name="encoder_conv_4")(encoder_activ_layer3)
        encoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
        encoder_activ_layer4 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_4")(encoder_norm_layer4)

        encoder_conv_layer5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", strides=1,
                                                             name="encoder_conv_5")(encoder_activ_layer4)
        encoder_norm_layer5 = tf.keras.layers.BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
        encoder_activ_layer5 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_5")(encoder_norm_layer5)

        shape_before_flatten = tf.keras.backend.int_shape(encoder_activ_layer5)[1:]
        encoder_flatten = tf.keras.layers.Flatten()(encoder_activ_layer5)

        encoder_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
        encoder_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(
            encoder_flatten)

        encoder_mu_log_variance_model = tf.keras.models.Model(x, (encoder_mu, encoder_log_variance),
                                                                      name="encoder_mu_log_variance_model")



        encoder_output = tf.keras.layers.Lambda(self.sampling, name="encoder_output")(
            [encoder_mu, encoder_log_variance])

        encoder = tf.keras.models.Model(x, encoder_output, name="encoder_model")


        ##################################################################################################
        # Decoder

        decoder_input = tf.keras.layers.Input(shape=(latent_space_dim), name="decoder_input")
        decoder_dense_layer1 = tf.keras.layers.Dense(units=np.prod(shape_before_flatten),
                                                             name="decoder_dense_1")(decoder_input)
        decoder_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)
        decoder_conv_tran_layer1 = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3,
                                                                           padding="same", strides=1,
                                                                           name="decoder_conv_tran_1")(decoder_reshape)
        decoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="decoder_norm_1")(
            decoder_conv_tran_layer1)
        decoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)

        decoder_conv_tran_layer2 = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3,
                                                                           padding="same", strides=2,
                                                                           name="decoder_conv_tran_2")(
            decoder_activ_layer1)
        decoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="decoder_norm_2")(
            decoder_conv_tran_layer2)
        decoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

        decoder_conv_tran_layer3 = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3,
                                                                           padding="same", strides=2,
                                                                           name="decoder_conv_tran_3")(
            decoder_activ_layer2)
        decoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="decoder_norm_3")(
            decoder_conv_tran_layer3)
        decoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

        decoder_conv_tran_layer4 = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=3,
                                                                           padding="same", strides=1,
                                                                           name="decoder_conv_tran_4")(
            decoder_activ_layer3)
        decoder_output = tf.keras.layers.LeakyReLU(name="decoder_output")(decoder_conv_tran_layer4)
        decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")

        vae_input = tf.keras.layers.Input(shape=(seq_length, vocab_size), name="VAE_input")
        vae_encoder_output = encoder(vae_input)
        vae_decoder_output = decoder(vae_encoder_output)
        vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")

        self.encoder = encoder
        self.decoder = decoder
        self.model = vae

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                           loss=self.loss_func(encoder_mu, encoder_log_variance))

    def loss_func(self, encoder_mu, encoder_log_variance):
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1000
            reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_predict),
                                                                axis=[1, 2, 3])
            return reconstruction_loss_factor * reconstruction_loss

        def vae_kl_loss(encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * tf.keras.backend.sum(
                1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(
                    encoder_log_variance), axis=1)
            return kl_loss

        def vae_kl_loss_metric(y_true, y_predict):
            kl_loss = -0.5 * tf.keras.backend.sum(
                1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(
                    encoder_log_variance), axis=1)
            return kl_loss

        def vae_loss(y_true, y_predict):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
            kl_loss = vae_kl_loss(y_true, y_predict)

            loss = reconstruction_loss + kl_loss
            return loss

        return vae_loss
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
        Hard values made from previous testing with VAE. These values may not be suitable for protein sequence
        generation
        """

        PathBuilder.build(result_path)

        self._get_ml_model(dataset.shape[1], dataset.shape[2])

        train_len = int(dataset.shape[0] * (2/3))
        x_train = dataset[:train_len]
        x_test = dataset[train_len:]

        enc = self.encoder(x_train[:2])
        print(self.decoder(enc))

        self.model.fit(x_train, x_train, epochs=20, batch_size=32, shuffle=True, validation_data=(x_test, x_test))

        self.encoder.save(result_path / "VAE_encoder.h5")
        self.decoder.save(result_path / "VAE_decoder.h5")
        self.model.save(result_path / "VAE.h5")



        return self.model

    def generate(self, amount=10, path_to_model: Path = None):

        self._get_ml_model(vocab_size=self.model_params['vocab_size'],
                           rnn_units=self.model_params['rnn_units'],
                           embedding_dim=self.model_params['embedding_dim'],
                           batch_size=1)

        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))

        sentence = "CARFL" #Random sequence chosen from dataset
        print('...Generating with seed: "' + sentence + '"')

        input_vect = [self.char2idx[s] for s in sentence]
        input_vect = tf.expand_dims(input_vect, 0)
        generated_seq = ''
        temperature = 1.
        self.model.reset_states()
        count = 0
        bar = pyprind.ProgBar(amount, bar_char="=", stream=sys.stdout, width=100)

        while True:
            prediction = self.model(input_vect)
            prediction = tf.squeeze(prediction, 0)
            prediction = prediction / temperature
            predicted_char = tf.random.categorical(prediction, num_samples=1)[-1, 0].numpy()
            input_vect = tf.expand_dims([predicted_char], 0)
            if self.idx2char[predicted_char] == ' ':
                count += 1
                bar.update()
            if count == amount:
                break
            generated_seq += self.idx2char[predicted_char]

        print()
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
        doc = str(VAE.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("VAE"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc