import os
import pickle

from scipy import sparse
from sklearn.preprocessing import StandardScaler, Normalizer

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.util.PathBuilder import PathBuilder


class FeatureScaler:

    SKLEARN_NORMALIZATION_TYPES = ["l1", "l2", "max"]

    @staticmethod
    def standard_scale(scaler_file: str, feature_matrix, with_mean: bool = True):
        """
        scale to zero mean and unit variance on feature level
        :param scaler_file: path to scaler file fitted on train set or where the resulting scaler file will be stored
        :param feature_matrix: rows -> examples, columns -> features
        :param with_mean: whether to scale to zero mean or not (could lose sparsity if scaled)
        :return: csc_matrix
        """

        if os.path.isfile(scaler_file):
            with open(scaler_file, 'rb') as file:
                scaler = pickle.load(file)
                scaled_feature_matrix = scaler.transform(feature_matrix)
        else:
            scaler = StandardScaler(with_mean=with_mean)
            scaled_feature_matrix = scaler.fit_transform(feature_matrix)

            PathBuilder.build(os.path.dirname(scaler_file))

            with open(scaler_file, 'wb') as file:
                pickle.dump(scaler, file)

        return sparse.csc_matrix(scaled_feature_matrix)

    @staticmethod
    def normalize(normalizer_filename: str, feature_matrix, normalization_type: NormalizationType):
        """
        normalize on example level so that the norm type applies
        :param normalizer_filename: where to store the normalizer
        :param feature_matrix: rows -> examples, columns -> features
        :param normalization_type: l1, l2, max
        :return: normalized feature matrix
        """
        if normalization_type.name == "NONE":
            normalized_feature_matrix = feature_matrix
        elif normalization_type.value in FeatureScaler.SKLEARN_NORMALIZATION_TYPES:
            normalized_feature_matrix = FeatureScaler._sklearn_normalize(normalizer_filename, feature_matrix, normalization_type)
        else:
            raise NotImplementedError("Normalization type {} ({}) has not yet been implemented.".format(normalization_type.name, normalization_type.value))

        return normalized_feature_matrix

    @staticmethod
    def _sklearn_normalize(normalizer_filename: str, feature_matrix, normalization_type: NormalizationType):
        if os.path.isfile(normalizer_filename):
            with open(normalizer_filename, 'rb') as file:
                normalizer = pickle.load(file)
                normalized_feature_matrix = normalizer.transform(feature_matrix)
        else:
            normalizer = Normalizer(norm=normalization_type.value)
            normalized_feature_matrix = normalizer.fit_transform(feature_matrix)

            PathBuilder.build(os.path.dirname(normalizer_filename))

            with open(normalizer_filename, 'wb') as file:
                pickle.dump(normalizer, file)
        return normalized_feature_matrix
