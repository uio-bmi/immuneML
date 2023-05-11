
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

from pathlib import Path
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.PathBuilder import PathBuilder
from tensorflow.python.framework.ops import disable_eager_execution


class VAE(GenerativeModel):
    """
    This is an implementation of Variational Autoencoders as a generative model.
    This ML method applies a collection of Keras Layers, specifically convolutional layers to produce a VAE. The
    structure of the model is based on the implementation by Hawkins-Hooker et al. (2021)
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008736

    For usage instructions, check :py:obj:`~immuneML.ml_methods.GenerativeModel.GenerativeModel`.

    Arguments specific for VAE:

        layers: number of convolutional layers to add in both encoding and decoding
        latent_dim: dimensionality of the latent layer
        batch_size: size of batches trained on at a time
        epochs: number of epochs to train the model


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_vae:
            VAE:
                # params
                layers: 2
                epochs: 2
                latent_dim: 2
                batch_size: 64
        # alternative way to define ML method with default values:
        my_default_vae: VAE

    """


    def get_classes(self) -> list:
        pass

    def __init__(self, **parameters):
        parameters = parameters if parameters is not None else {}
        super(VAE, self).__init__(parameters=parameters)
        disable_eager_execution()

        self._latent_dim = self._parameters['latent_dim']
        self._epochs = self._parameters["epochs"]
        self._batch_size = self._parameters["batch_size"]
        self._layers = self._parameters['layers']

        self.historydf = {}
        self.encoder = None
        self.decoder = None
        self.generated_sequences = []

    def _get_encoder(self, seq_length, vocab_size, latent_space, layers,
                     num_filters=21, max_filters=10000, kernel_size=2):
        encoder_input = tf.keras.layers.Input((seq_length, vocab_size), name="encoder_input")
        e = encoder_input
        for i in range(layers):
            e = tf.keras.layers.Conv1D(min(num_filters * (2 ** i), max_filters), kernel_size,
                                       strides=1 if i == 0 else 2, name='Conv1D_' + str(i))(e)
            e = tf.keras.layers.BatchNormalization(momentum=0.9)(e)
            e = tf.keras.layers.Dense(e.shape[2], activation='PReLU')(e)

        intermediate_dim = e.shape[1]
        flat = tf.keras.layers.Flatten(name='h_flat')(e)
        z_mean = tf.keras.layers.Dense(latent_space, name='z_mean')(flat)
        z_var = tf.keras.layers.Dense(latent_space, activation='softplus', name='z_var')(flat)
        z = tf.keras.layers.Lambda(self.sampler)([z_mean, z_var])

        self.encoder = tf.keras.models.Model(encoder_input, [z_mean, z_var, z])
        return encoder_input, intermediate_dim

    def _get_decoder(self, seq_length, vocab_size, latent_space, layers,
                     intermediate_dim=63, num_filters=42, max_filters=336, kernel_size=2):
        latent_vector = tf.keras.layers.Input((latent_space,), name='latent_input')

        prot_oh = tf.keras.layers.Input((seq_length, vocab_size), name='protein_input')
        input_x = tf.keras.layers.ZeroPadding1D(padding=(1, 0))(prot_oh)
        input_x = tf.keras.layers.Lambda(lambda x_: x_[:, :-1, :])(input_x)

        input_x = tf.keras.layers.Dropout(0.45, noise_shape=(None, seq_length, 1))(input_x)

        input_x = tf.keras.layers.Conv1D(vocab_size, 1, activation=None, name='decoder_x_embed')(input_x)

        rnn = tf.keras.layers.GRU(512, return_sequences=True)

        low_res_features = min(num_filters * (2 ** layers), max_filters)
        d = tf.keras.layers.Dense(intermediate_dim * low_res_features, name='upsampler_mlp')(latent_vector)
        d = tf.keras.layers.Reshape((intermediate_dim, low_res_features))(d)

        for i in range(layers):
            filters = min(num_filters * 2 ** (layers - (i + 1)), max_filters)
            d = tf.keras.layers.Conv1DTranspose(filters,
                                                kernel_size=kernel_size,
                                                strides=2,
                                                padding='same',
                                                dilation_rate=1,
                                                activation=None,
                                                use_bias=False,
                                                )(d)
            d = tf.keras.layers.BatchNormalization(momentum=0.9)(d)

            d = tf.keras.layers.Dense(d.shape[2], activation='PReLU')(d)

        z_seq = tf.keras.layers.Flatten()(d)
        z_seq = tf.keras.layers.Dense(seq_length * d.shape[2])(z_seq)
        z_seq = tf.keras.layers.Reshape((seq_length, d.shape[2]), name='Reshaped_upsampled')(z_seq)

        xz_seq = tf.keras.layers.Concatenate(axis=-1, name='concatinated')([z_seq, input_x])
        rnn_out = rnn(xz_seq)

        processed_x = tf.keras.layers.Conv1D(vocab_size, 1, activation=None, use_bias=True, name='processed_x')(rnn_out)
        output = tf.keras.layers.Dense(processed_x.shape[-1], activation='softmax')(processed_x)

        self.decoder = tf.keras.models.Model([latent_vector, prot_oh], output)

    def _get_ml_model(self, seq_length, vocab_size, latent_space=2, layers=4, num_filters=21,
                     max_filters=336, kernel_size=3, min_deconv_dim=21):

        encoder_input, intermediate_dim = self._get_encoder(seq_length, vocab_size, latent_space, layers)
        self._get_decoder(seq_length, vocab_size, latent_space, layers, intermediate_dim)
        z_mean, z_var, z = self.encoder(encoder_input)
        decoder_output = self.decoder([z, encoder_input])

        self.model = tf.keras.models.Model(encoder_input, decoder_output, name="VAE")

        self.model.compile(loss=self.loss_func(z_mean, z_var),
                           optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001))

    @staticmethod
    def sampler(zs):
        z_mean, z_var = zs
        epsilon = K.sqrt(tf.convert_to_tensor(z_var + 1e-8, np.float32)) * \
                  K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1)

        return z_mean + epsilon

    @staticmethod
    def loss_func(z_mean, z_var):

        def xent_loss(x, x_d_m):
            xent = K.sum(tf.keras.losses.categorical_crossentropy(x, x_d_m), -1)
            return xent

        def kl_loss():
            kl = - 0.5 * K.sum(1 + K.log(z_var + 1e-8) - K.square(z_mean) - z_var, axis=-1)
            return kl

        def vae_loss(x, x_d_m):
            return xent_loss(x, x_d_m) + kl_loss()

        return vae_loss

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

        train_len = int(one_hot.shape[0] / 2)
        x_train = one_hot[:train_len]
        x_test = one_hot[train_len:train_len + int((train_len / 2))]
        x_val = one_hot[train_len + int((train_len / 2)):]

        self._get_ml_model(one_hot.shape[1], one_hot.shape[2], latent_space=self._latent_dim, layers=self._layers)

        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

        history = self.model.fit(x_train,
                       x_train,
                       epochs=self._epochs,
                       batch_size=self._batch_size,
                       shuffle=True,
                       validation_data=(x_test,
                                        x_test),
                       workers=cores_for_training)

        history_contents = []
        for metric in history.history:
            metric_contents = []
            for i, val in enumerate(history.history[metric]):
                history_content = [val]
                metric_contents.append(history_content)
            history_contents.append([metric, metric_contents])
        self.historydf = pd.DataFrame(history_contents, columns=['metric', 'data'])

        # If latent_space = 2, this can be used to visualize the latent space after encoding
        # latent = np.transpose(self.encoder.predict(x_val)[2])
        #
        # x = latent[0]
        # y = latent[1]
        #
        # plt.scatter(x, y)
        # plt.show()

        return self.model

    def generate(self):

        z = np.random.randn(self._amount, self.decoder.input_shape[0][1])

        original_dim, alphabet_size = self.decoder.output_shape[1], self.decoder.output_shape[-1]
        x = np.zeros((z.shape[0], original_dim, alphabet_size))
        start = 0

        for i in range(start, original_dim):
            # iteration is over positions in sequence, which can't be parallelized
            pred = self.decoder.predict([z, x])
            pos_pred = pred[:, i, :]
            pred_ind = np.argmax(pos_pred, -1)  # convert probability to index
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
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder

        return [OneHotEncoder]

    def load(self, path: Path, details_path: Path = None):
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
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.GenerativeModel.GenerativeModel`.": GenerativeModel.get_usage_documentation("VAE"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc
