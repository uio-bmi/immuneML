import abc

from source.data_model.encoded_data.EncodedData import EncodedData


class MLMethod(metaclass=abc.ABCMeta):
    """
        Base class for wrappers for different ML/DL methods.
    """
    def __init__(self):
        self.ml_details_path = None
        self.predictions_path = {}

    @abc.abstractmethod
    def fit(self, encoded_data: EncodedData, y, label_names: list = None, cores_for_training: int = 2):
        pass

    @abc.abstractmethod
    def predict(self, encoded_data: EncodedData, label_names: list = None):
        pass

    @abc.abstractmethod
    def fit_by_cross_validation(self, encoded_data: EncodedData, y, number_of_splits: int = 5, parameter_grid: dict = None,
                                label_names: list = None):
        pass

    @abc.abstractmethod
    def store(self, path, feature_names=None, details_path=None):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def get_model(self, label_names: list = None):
        pass

    @abc.abstractmethod
    def check_if_exists(self, path):
        pass

    @abc.abstractmethod
    def get_classes_for_label(self, label):
        pass

    @abc.abstractmethod
    def get_params(self, label):
        pass

    @abc.abstractmethod
    def predict_proba(self, encoded_data: EncodedData, labels):
        pass

    @abc.abstractmethod
    def get_labels(self):
        pass
