import pandas as pd
import numpy as np
from scipy.sparse import issparse
import torch


class EncodedData:
    """
    When a dataset is encoded, it is stored in an object of EncodedData class.

    Arguments:

      examples: a design matrix containing the encoded data. This is typically a numpy array, although other matrix formats such as scipy sparse matrix, pandas dataframe
                or pytorch tensors are also permitted as long as the numpy matrix can be retrieved using 'get_examples_as_np_matrix()'.
                The matrix is usually two-dimensional. The first dimension should be the examples, and the second (and higher) dimensions represent features.


      feature_names: a list of feature names. The length (dimensions) of this list should match the number of features in the examples matrix.

      feature_annotations: a data frame consisting of additional annotations for each feature. This can be used to add more information fields if feature_names is not sufficient. This data field is not used for machine learning, but may be used by some Reports.

      example_ids: a list of example (repertoire/sequence/receptor) IDs; it must be the same length as the example_count in the examples matrix. These can be retrieved using Dataset.get_example_ids()

      labels: a dict of labels where label names are keys and the values are lists of values for the label across examples: {'disease1': ['sick', 'healthy', 'sick']}
              During encoding, the labels can be computed using EncoderHelper.encode_dataset_labels()
      """

    def __init__(self, examples, labels: dict = None, example_ids: list = None, feature_names: list = None,
                 feature_annotations: pd.DataFrame = None, encoding: str = None, example_weights: list = None, info: dict = None,
                 dimensionality_reduced_data: np.ndarray = None):

        assert feature_names is None or examples.shape[1] == len(feature_names), f"EncodedData: the length of feature_names ({len(feature_names)}) must match the feature dimension of the example matrix ({examples.shape[1]})"
        if feature_names is not None:
            assert feature_annotations is None or feature_annotations.shape[0] == len(feature_names) == examples.shape[1]
        if example_ids is not None and labels is not None:
            for label in labels.values():
                assert len(label) == len(example_ids), "EncodedData: there are {} labels, but {} examples"\
                    .format(len(label), len(example_ids))
                assert examples is None or len(example_ids) == examples.shape[0], "EncodedData: there are {} example ids, but {} examples."\
                    .format(len(example_ids), examples.shape[0])

            if example_weights is not None:
                assert len(example_weights) == len(example_ids)
        if examples is not None:
            assert all(len(labels[key]) == examples.shape[0] for key in labels.keys()) if labels is not None else True

            if example_weights is not None:
                assert len(example_weights) == examples.shape[0]

        self.examples = examples
        self.labels = labels
        self.example_ids = example_ids
        self.feature_names = feature_names
        self.feature_annotations = feature_annotations
        self.encoding = encoding
        self.example_weights = example_weights
        self.info = info
        self.dimensionality_reduced_data = dimensionality_reduced_data

    def get_examples_as_np_matrix(self):
        if isinstance(self.examples, np.ndarray):
            return self.examples
        elif isinstance(self.examples, pd.DataFrame):
            return self.examples.to_numpy()
        elif issparse(self.examples):
            return self.examples.toarray()
        elif torch.is_tensor(self.examples):
            return self.examples.numpy()
        else:
            raise ValueError(f"EncodedData: examples matrix of type '{type(self.examples)}' cannot be converted to a numpy matrix.")
