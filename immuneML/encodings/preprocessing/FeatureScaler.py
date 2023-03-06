from sklearn.preprocessing import normalize, binarize
import numpy as np

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType


class FeatureScaler:

    SKLEARN_NORMALIZATION_TYPES = ["l1", "l2", "max"]

    @staticmethod
    def standard_scale_fit(scaler, design_matrix, with_mean: bool = True):
        """
        Scale to zero mean and unit variance on feature level

        Args:

           scaler: scaler object that has function fit_transform

           design_matrix: rows -> examples, columns -> features

           with_mean: whether to scale to zero mean or not (could lose sparsity if scaled)

        Returns:

            scaled design matrix
        """

        scaled_design_matrix = FeatureScaler._optional_convert_to_dense(design_matrix, with_mean)
        scaled_design_matrix = scaler.fit_transform(scaled_design_matrix)

        return scaled_design_matrix

    @staticmethod
    def standard_scale(scaler, design_matrix, with_mean: bool = True):
        """
        Scale to zero mean and unit variance on feature level

        Args:

           scaler: already fitted scaler object that has function transform

           design_matrix: rows -> examples, columns -> features

           with_mean: whether to scale to zero mean or not (could lose sparsity if scaled)

        Returns:

            scaled design matrix
        """

        scaled_design_matrix = FeatureScaler._optional_convert_to_dense(design_matrix, with_mean)
        scaled_design_matrix = scaler.transform(scaled_design_matrix)

        return scaled_design_matrix

    @staticmethod
    def _optional_convert_to_dense(design_matrix, with_mean: bool):
        if with_mean and hasattr(design_matrix, "todense"):
            scaled_design_matrix = np.array(design_matrix.todense())
        else:
            scaled_design_matrix = design_matrix
        return scaled_design_matrix

    @staticmethod
    def normalize(design_matrix, normalization_type: NormalizationType):
        """
        Normalize on example level so that the norm type applies to compute values like frequency

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
