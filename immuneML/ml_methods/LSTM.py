import keras
import tensorflow as tf
import numpy as np
from keras.layers.rnn.lstm import LSTM as LongShortTermMemory

from immuneML.ml_methods.GenerativeModel import GenerativeModel
from scripts.specification_util import update_docs_per_mapping


class LSTM(GenerativeModel):
    """
    This is a wrapper of scikit-learn’s KNeighborsClassifier class.
    This ML method creates a distance matrix using the given encoded data. If the encoded data is already a distance
    matrix (for example, when using the :ref:`Distance` or :ref:`CompAIRRDistance` encoders), please use :ref:`PrecomputedKNN` instead.

    Please see the `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
    of KNeighborsClassifier for the parameters.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_knn_method:
            KNN:
                # sklearn parameters (same names as in original sklearn class)
                weights: uniform # always use this setting for weights
                n_neighbors: [5, 10, 15] # find the optimal number of neighbors
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under KNN is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_knn: KNN

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = parameters if parameters is not None else {}
        parameter_grid = parameter_grid if parameter_grid is not None else {}

        super(LSTM, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        params = self._parameters
        print("dette er params", params)

        vocab_size = 21
        embedding_dim = 256
        batch_size = 64

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            LongShortTermMemory(**params),
            tf.keras.layers.Dense(vocab_size)
        ])

        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())

        return model


    def get_params(self):
        return self.model.get_params(deep=True)

    def can_predict_proba(self) -> bool:
        return True

    def get_compatible_encoders(self):
        from immuneML.encodings.distance_encoding.DistanceEncoder import DistanceEncoder
        from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
        from immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
        from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import CompAIRRSequenceAbundanceEncoder
        from immuneML.encodings.abundance_encoding.KmerAbundanceEncoder import KmerAbundanceEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
        return [KmerFrequencyEncoder, OneHotEncoder, Word2VecEncoder, SequenceAbundanceEncoder,
                CompAIRRSequenceAbundanceEncoder, KmerAbundanceEncoder, EvennessProfileEncoder, DistanceEncoder]

    @staticmethod
    def get_documentation():
        doc = str(LSTM.__doc__)

        mapping = {
            "For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.": GenerativeModel.get_usage_documentation("LSTM"),
        }

        doc = update_docs_per_mapping(doc, mapping)
        return doc
