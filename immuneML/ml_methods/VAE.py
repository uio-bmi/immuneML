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

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0,
                                                 stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
        return random_sample

    def _get_ml_model(self, seq_length, vocab_size, latent_space_dim=2):

        # Encoder
        x = tf.keras.layers.Input(shape=(seq_length, vocab_size), name="encoder_input")

        encoder_conv_layer1 = tf.keras.layers.Dense(64)(x)
        encoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
        encoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)

        encoder_conv_layer2 = tf.keras.layers.Dense(32)(encoder_activ_layer1)
        encoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
        encoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_2")(encoder_norm_layer2)

        encoder_conv_layer3 = tf.keras.layers.Dense(16)(encoder_activ_layer2)
        encoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
        encoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_3")(encoder_norm_layer3)

        encoder_conv_layer4 = tf.keras.layers.Dense(8)(encoder_activ_layer3)
        encoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
        encoder_activ_layer4 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_4")(encoder_norm_layer4)

        encoder_conv_layer5 = tf.keras.layers.Dense(8)(encoder_activ_layer4)
        encoder_norm_layer5 = tf.keras.layers.BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
        encoder_activ_layer5 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_5")(encoder_norm_layer5)

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
        decoder_dense_layer1 = tf.keras.layers.Dense(units=np.prod((seq_length, 8)))(decoder_input)

        decoder_reshape = tf.keras.layers.Reshape(target_shape=(seq_length, 8))(decoder_dense_layer1)
        decoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="decoder_norm_1")(
            decoder_reshape)
        decoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)

        decoder_dense_layer2 = tf.keras.layers.Dense(16)(decoder_activ_layer1)
        decoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="decoder_norm_2")(
            decoder_dense_layer2)
        decoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

        decoder_dense_layer3 = tf.keras.layers.Dense(32)(decoder_activ_layer2)
        decoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="decoder_norm_3")(
            decoder_dense_layer3)
        decoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

        flatten = tf.keras.layers.Flatten()(decoder_activ_layer3)
        decoder_dense_layer4 = tf.keras.layers.Dense(np.prod((seq_length, vocab_size)))(flatten)

        decoder_reshape_4 = tf.keras.layers.Reshape(target_shape=(seq_length, vocab_size))(decoder_dense_layer4)

        decoder_output = tf.keras.layers.Dense(vocab_size, activation="softmax", name="decoder_output")(decoder_reshape_4)



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
            reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_predict))
            return reconstruction_loss_factor #* reconstruction_loss

        def vae_kl_loss(encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * tf.keras.backend.sum(
                1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(
                    encoder_log_variance))
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

    def _fit(self, dataset, cores_for_training: int = 1, result_path: Path = None):

        """
        Hard values made from previous testing with VAE. These values may not be suitable for protein sequence
        generation
        """

        sequences_dict = {}
        output_figures = []
        dataset = dataset.split(" ")
        for sequence in dataset:
            if len(sequence) not in sequences_dict.keys():
                sequences_dict[len(sequence)] = [sequence]
            else:
                sequences_dict[len(sequence)].append(sequence)

        new_sequences = []
        for key in sequences_dict:
            if len(sequences_dict[key]) > len(new_sequences):
                new_sequences = sequences_dict[key]

        one_hot = np.zeros((len(new_sequences), len(new_sequences[1]), len(self.alphabet)))

        new_sequences_int = []
        for sequence in new_sequences:
            new_sequences_int.append([self.char2idx[i] for i in sequence])
        #one_hot[np.arange((len(new_sequences), len(new_sequences[1])), new_sequences_int)] = 1

        one_hot = tf.one_hot(new_sequences_int, len(self.alphabet))

        PathBuilder.build(result_path)

        self._get_ml_model(one_hot.shape[1], one_hot.shape[2])

        train_len = int(one_hot.shape[0] * (2/3))
        x_train = one_hot[:train_len]
        x_test = one_hot[train_len:]

        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

        self.model.fit(x_train, x_train, epochs=20, batch_size=32, shuffle=True, validation_data=(x_test, x_test), workers=cores_for_training)

        return self.model

    def generate(self, amount=10, path_to_model: Path = None):

        #Consider different way of getting latent variables
        fake_latent = np.random.rand(amount, 1, 2)

        gens = []

        for i in fake_latent:
            decoded = self.decoder([i])
            maxed = tf.math.argmax(decoded, axis=2)
            char_seq = ""
            for seq in maxed[0]:
                char_seq = char_seq + self.idx2char[seq]
            gens.append(char_seq)
        self.generated_sequences = gens

        return gens

    def get_params(self):
        return self._parameters


    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        self.model = tf.keras.models.load_model(path / "VAE", custom_objects={"sampling":self.sampling}, compile=False)
        self.encoder = tf.keras.models.load_model(path / "VAE_encoder", custom_objects={"sampling":self.sampling})
        self.decoder = tf.keras.models.load_model(path / "VAE_decoder")


    def store(self, path: Path, feature_names=None, details_path: Path = None):

        PathBuilder.build(path)

        self.encoder.save(path / "VAE_encoder")
        self.decoder.save(path / "VAE_decoder")
        self.model.save(path / "VAE")

    @staticmethod
    def get_documentation():
        doc = str(VAE.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("VAE"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc