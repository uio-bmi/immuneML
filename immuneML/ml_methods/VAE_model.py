import keras
import keras.backend as K
import tensorflow as tf


class VAE_model(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE_model, self).__init__(**kwargs)
        self.e = encoder
        self.d = decoder

    def call(self, inputs, training=None, mask=None):
        _,_,z = self.e(inputs)
        prot = self.e.inputs[0]
        return self.d([z, prot])
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.e(data)
            prot = self.e.inputs[0]
            predicted = self.d([z, prot])
            xent_loss = K.sum(tf.keras.losses.categorical_crossentropy(data, predicted), -1) / 100

            kl_loss = - 0.5 * K.sum(1 + K.log(z_log_var + 1e-8) - K.square(z_mean) - z_log_var, axis=-1)
            # beta =10
            total_loss = xent_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "xent_loss": xent_loss,
            "kl_loss": kl_loss,
        }
