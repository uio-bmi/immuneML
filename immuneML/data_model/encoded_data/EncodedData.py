import pandas as pd


class EncodedData:
    """
    When a dataset is encoded, it is stored in an object of EncodedData class.

    Arguments:
      examples: a matrix of example_count x feature_count elements (can be a numpy array or a sparse matrix); there are some exceptions to this, for
        instance, :py:obj:`source.encodings.onehot.OneHotEncoder.OneHotEncoder` where the numpy array has more than two dimensions, but most of the
        encodings follow the matrix format.
      feature_names: a list of feature names with feature_count elements
      feature_annotations: a data frame consisting of annotations for each unique feature
      example_ids: a list of example (repertoire/sequence/receptor) IDs; it must be the same length as the example_count in the examples matrix
      labels: a dict of labels where label names are keys and the values are lists of values for the label across examples: {label_name1: [...], label_name2: [...]}. Each list associated with a label has to have values for all examples.

    """

    def __init__(self, examples, labels: dict = None, example_ids: list = None, feature_names: list = None,
                 feature_annotations: pd.DataFrame = None, encoding: str = None, info: dict = None):

        assert feature_names is None or examples.shape[1] == len(feature_names)
        if feature_names is not None:
            assert feature_annotations is None or feature_annotations.shape[0] == len(feature_names) == examples.shape[1]
        if example_ids is not None and labels is not None:
            for label in labels.values():
                assert len(label) == len(example_ids), "EncodedData: there are {} labels, but {} examples"\
                    .format(len(label), len(example_ids))
                assert examples is None or len(example_ids) == examples.shape[0], "EncodedData: there are {} example ids, but {} examples."\
                    .format(len(example_ids), examples.shape[0])
        if examples is not None:
            assert all(len(labels[key]) == examples.shape[0] for key in labels.keys()) if labels is not None else True

        self.examples = examples
        self.labels = labels
        self.example_ids = example_ids
        self.feature_names = feature_names
        self.feature_annotations = feature_annotations
        self.encoding = encoding
        self.info = info
