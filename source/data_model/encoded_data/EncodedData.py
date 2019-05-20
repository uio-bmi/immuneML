class EncodedData:
    """
    When a dataset is encoded, it is stored in an object of EncodedData class;

    It consists of:
        repertoires: a matrix of repertoire_count x feature_count elements
        feature_names: a list of feature names with feature_count elements
        repertoire_ids: a list of repertoire IDs with repertoire_count elements
        labels: a dict of labels where each label is a key and the value is a list of values
                for the label across repertoires:
                {label_name1: [...], label_name2: [...]}
                Each list associated with a label has to have values for all repertories
    """

    def __init__(self, repertoires, labels: dict, repertoire_ids: list = None, feature_names: list = None):

        assert feature_names is None \
               or all(repertoires[i].shape[0] == len(feature_names) for i in range(repertoires.shape[0]))
        assert len(labels.keys()) > 0
        assert all(len(labels[key]) == repertoires.shape[0] for key in labels.keys())

        self.repertoires = repertoires
        self.repertoire_ids = repertoire_ids
        self.labels = labels
        self.feature_names = feature_names
