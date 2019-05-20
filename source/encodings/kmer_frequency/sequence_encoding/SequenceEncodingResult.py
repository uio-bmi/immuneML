class SequenceEncodingResult:

    def __init__(self, features, feature_information_names):
        self.features = features
        self.feature_information_names = feature_information_names

    def __eq__(self, other):
        return self.feature_information_names == other.feature_information_names and self.features == other.features
