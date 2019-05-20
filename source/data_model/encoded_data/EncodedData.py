import pandas as pd

class EncodedData:
    """
    When a dataset is encoded, it is stored in an object of EncodedData class;

    It consists of:
        repertoires: a matrix of repertoire_count x feature_count elements
        feature_names: a list of feature names with feature_count elements
        feature_annotations: a data frame consisting of annotations for each unique feature
        repertoire_ids: a list of repertoire IDs with repertoire_count elements
        labels: a dict of labels where each label is a key and the value is a list of values
                for the label across repertoires:
                {label_name1: [...], label_name2: [...]}
                Each list associated with a label has to have values for all repertories
    """

    def __init__(self, repertoires, labels: dict, repertoire_ids: list = None, feature_names: list = None, feature_annotations: pd.DataFrame = None):

        assert feature_names is None \
               or repertoires.shape[1] == len(feature_names)
        if feature_names is not None:
            assert feature_annotations is None or feature_annotations.shape[0] == len(feature_names) == repertoires.shape[1]
        assert len(labels.keys()) > 0
        assert all(len(labels[key]) == repertoires.shape[0] for key in labels.keys())

        self.repertoires = repertoires
        self.repertoire_ids = repertoire_ids
        self.labels = labels
        self.feature_names = feature_names
        self.feature_annotations = feature_annotations
