import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler, normalize, binarize

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.util.PathBuilder import PathBuilder


class FeatureScaler:

    SKLEARN_NORMALIZATION_TYPES = ["l1", "l2", "max"]

    @staticmethod
    def standard_scale(scaler_file: Path, design_matrix, with_mean: bool = True):
        """
        scale to zero mean and unit variance on feature level
        :param scaler_file: path to scaler file fitted on train set or where the resulting scaler file will be stored
        :param design_matrix: rows -> examples, columns -> features
        :param with_mean: whether to scale to zero mean or not (could lose sparsity if scaled)
        :return: scaled design matrix
        """

        if with_mean and hasattr(design_matrix, "todense"):
            scaled_design_matrix = design_matrix.todense()
        else:
            scaled_design_matrix = design_matrix

        if scaler_file.is_file():
            with scaler_file.open('rb') as file:
                scaler = pickle.load(file)
                scaled_design_matrix = scaler.transform(scaled_design_matrix)
        else:
            scaler = StandardScaler(with_mean=with_mean)
            scaled_design_matrix = scaler.fit_transform(scaled_design_matrix)

            directory = scaler_file.parent
            PathBuilder.build(directory)

            with scaler_file.open('wb') as file:
                pickle.dump(scaler, file)

        return scaled_design_matrix

    @staticmethod
    def normalize(design_matrix, normalization_type: NormalizationType):
        """
        Normalize on example level so that the norm type applies

        Args:
            design_matrix: rows -> examples, columns -> features
            normalization_type: l1, l2, max, binary, none

        Returns:
             normalized design matrix
        """
        if normalization_type.name == "NONE":
            normalized_design_matrix = design_matrix
        elif normalization_type.name == "BINARY":
            normalized_design_matrix = binarize(design_matrix)
        elif normalization_type.value in FeatureScaler.SKLEARN_NORMALIZATION_TYPES:
            normalized_design_matrix = normalize(design_matrix, norm=normalization_type.value, axis=1)
        else:
            raise NotImplementedError("Normalization type {} ({}) is not implemented.".format(normalization_type.name, normalization_type.value))

        return normalized_design_matrix
