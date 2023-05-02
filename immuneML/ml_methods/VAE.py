import random

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder
from tensorflow.python.framework.ops import disable_eager_execution
from immuneML.ml_methods.VAE_model import VAE_model


class VAE(GenerativeModel):

    def get_classes(self) -> list:
        pass

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}
        super(VAE, self).__init__(parameter_grid=parameter_grid, parameters=parameters)
        # disable_eager_execution()

        self.latent_dim = self._parameters["latent_dim"]
        self.layers = self._parameters["layers"]
        self.epochs = self._parameters["epochs"]
        self.batch_size = self._parameters["batch_size"]

        self.encoder = None
        self.decoder = None
        self.generated_sequences = []



    def sampler(self, latent_dim, epsilon_std=1):
        _sampling = lambda z_args: (z_args[0] + K.sqrt(z_args[1] + 1e-8) *
                                    K.random_normal(shape=K.shape(z_args[0]), mean=0., stddev=epsilon_std))

        return tf.keras.layers.Lambda(_sampling, output_shape=(latent_dim,))

    def _get_encoder(self, seq_length, vocab_size, latent_space, layers=4, num_filters=21,
                     max_filters=10000, kernel_size=3):

        x = tf.keras.layers.Input((seq_length, vocab_size), name="encoder_input")
        h = x
        for i in range(layers):
            h = tf.keras.layers.Conv1D(min(num_filters*(2**i), max_filters), kernel_size,
                                       strides=1)(h)
            h = tf.keras.layers.BatchNormalization(momentum=0.9)(h)
            h = tf.keras.layers.Dense(h.shape[-1], activation='PReLU')(h)

        h = tf.keras.layers.Flatten()(h)

        z_mean = tf.keras.layers.Dense(latent_space)(h)
        z_var = tf.keras.layers.Dense(latent_space, activation='softplus')(h)
        z = self.sampler(latent_space)([z_mean, z_var])

        encoder = tf.keras.models.Model(x, [z_mean, z_var, z])

        return encoder

    def _get_decoder(self, seq_length, vocab_size, latent_space=2, ncell=512, project_x=True,
                               upsample=True, min_deconv_dim=42,
                               input_dropout=0.45, intermediate_dim=63,
                               max_filters=336, layers=3):
        latent_vector = tf.keras.layers.Input((latent_space,))

        prot_oh = tf.keras.layers.Input((seq_length, vocab_size))
        input_x = tf.keras.layers.ZeroPadding1D(padding=(1,0))(prot_oh)
        input_x = tf.keras.layers.Lambda(lambda x_: x_[:,:-1,:])(input_x)

        if input_dropout is not None:
            input_x = tf.keras.layers.Dropout(input_dropout, noise_shape=(None, seq_length, 1))(input_x)
        if project_x:
            input_x = tf.keras.layers.Conv1D(vocab_size, 1, activation=None, name='decoder_x_embed')(input_x)

        rnn = tf.keras.layers.GRU(ncell, return_sequences=True)
        if upsample:
            z_seq = self.upsampler(latent_vector, intermediate_dim, min_deconv_dim=min_deconv_dim,
                              layers=layers, max_filters=max_filters)
        else:
            z_seq = tf.keras.layers.RepeatVector(seq_length)(latent_vector)

        z_seq = tf.keras.layers.Flatten()(z_seq)
        z_seq = tf.keras.layers.Dense(seq_length*vocab_size)(z_seq)
        z_seq = tf.keras.layers.Reshape((seq_length, vocab_size))(z_seq)
        xz_seq = tf.keras.layers.Concatenate(axis=-1)([z_seq, input_x])
        rnn_out = rnn(xz_seq)

        processed_x = tf.keras.layers.Conv1D(vocab_size, 1, activation=None, use_bias=True)(rnn_out)
        output = tf.keras.layers.Activation('softmax')(processed_x)

        decoder = tf.keras.models.Model([latent_vector, prot_oh], output)
        return decoder

    def upsampler(self, latent_vector, low_res_dim, min_deconv_dim=21,
                  layers=4, kernel_size=3, max_filters=336, aim=28):

        low_res_features = min(min_deconv_dim * (2 ** layers), max_filters)
        h = tf.keras.layers.Dense(low_res_dim * low_res_features, name='upsampler_mlp')(latent_vector)
        h = tf.keras.layers.Reshape((low_res_dim, low_res_features))(h)

        for i in range(layers):
            shape = list(h.shape[1:])
            new_shape = shape[:-1] + [1] + [shape[-1]]
            h = tf.keras.layers.Reshape(new_shape)(h)
            filters = min(min_deconv_dim * 2 ** (layers - (i + 1)), max_filters)
            h = tf.keras.layers.Conv2DTranspose(filters,
                                kernel_size,
                                strides=(2, 1),
                                padding='same',
                                dilation_rate=1,
                                activation=None,
                                use_bias=False,
                                )(h)
            h = tf.keras.layers.BatchNormalization(momentum=0.9)(h)

            shape = list(h.shape[1:])
            new_shape = shape[:-2] + [filters]
            h = tf.keras.layers.Reshape(new_shape)(h)

            h = tf.keras.layers.Dense(h.shape[2], activation='PReLU')(h)

        return h


    def _get_ml_model(self, latent_dim, one_hot):

        self.encoder = self._get_encoder(one_hot.shape[1], one_hot.shape[2], latent_space=self.latent_dim, layers=self.layers)
        self.decoder = self._get_decoder(one_hot.shape[1], one_hot.shape[2], latent_dim, layers=self.layers)

        self.model = VAE_model(self.encoder, self.decoder)

        self.model.compile(loss='mse', optimizer='adam')


    def _fit(self, dataset, cores_for_training: int = 1, result_path: Path = None):

        dataset = np.insert(dataset, 0, 0, axis=2)

        new_data = []
        for data in dataset:
            index = 0
            for i, hot in enumerate(data):
                if sum(hot) == 0:
                    index = i
                    break
            converted_data = data[:int(index/2)]
            z = np.zeros((data.shape[1], (data.shape[0] - index)))
            z[0] = np.ones((data.shape[0] - index))
            converted_data = np.append(converted_data, z.T, axis=0)
            converted_data = np.append(converted_data, data[int(index/2):index], axis=0)
            new_data.append(converted_data)

        one_hot = np.array(new_data)

        self._get_ml_model(self.latent_dim, one_hot)

        train_len = int(one_hot.shape[0] / 2)
        x_train = one_hot[:train_len - (train_len % self.batch_size)]
        x_test = one_hot[train_len:train_len + int((train_len / 2))]
        x_val = one_hot[train_len + int((train_len / 2)):]

        self.encoder.summary()
        self.decoder.summary()
        # self.model.summary()

        self.model.fit(x_train,
                       x_train,
                       epochs=self.epochs,
                       # verbose=0,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=(x_test,
                                        x_test),
                       workers=cores_for_training)

        return self.model

    def generate(self, amount=10, mean=0, stddev=1, path_to_model: Path = None):

        z = mean + stddev * np.random.randn(amount, self.latent_dim)
        # x, y = z.transpose()
        # plt.scatter(x, y)
        # plt.show()
        print("Generating " + str(amount) + " sequences")
        original_dim, alphabet_size = self.decoder.output_shape[1], self.decoder.output_shape[-1]
        x = np.zeros((z.shape[0], original_dim, alphabet_size))

        for i in range(0, original_dim):
            # iteration is over positions in sequence, which can't be parallelized
            pred = self.decoder(z)
            pos_pred = pred[:, i, :]
            pred_ind = tf.argmax(pos_pred, -1)  # convert probability to index
            for j, p in enumerate(pred_ind):
                x[j, i, p] = 1

        sequences = []
        for sequence in x:
            char_sequence = ""
            for position in sequence:
                if position.argmax() != 0:
                    char_sequence += self.alphabet[position.argmax()-1]
            sequences.append(char_sequence)

        return sequences

    def get_params(self):
        return self._parameters


    def can_predict_proba(self) -> bool:
        raise Exception("can_predict_proba has not been implemented")

    def get_compatible_encoders(self):
        raise Exception("get_compatible_encoders has not been implemented")

    def load(self, path: Path, details_path: Path = None):

        self.model = tf.keras.models.load_model(path / "VAE", custom_objects={"sampling": self.sampling}, compile=False)
        self.encoder = tf.keras.models.load_model(path / "VAE_encoder", custom_objects={"sampling": self.sampling})
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
