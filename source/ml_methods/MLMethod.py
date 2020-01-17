import abc


class MLMethod(metaclass=abc.ABCMeta):
    """
        Base class for wrappers for different ML/DL methods
        adapted to work with RepertoireDataset objects and use their encoded_data
        attribute to access the data;

        These classes (MLMethod and subclasses) assume that encoded_data has
        the following format: {"repertoires" : scipy.sparse.csc_matrix, "labels": np.ndarray}

        Other parameters in addition to repertoires and labels can be included,
        such as label_names, features names etc.
    """
    def __init__(self):
        self.ml_details_path = None
        self.predictions_path = {}

    @abc.abstractmethod
    def fit(self, X, y, label_names: list = None, cores_for_training: int = 2):
        pass

    @abc.abstractmethod
    def predict(self, X, label_names: list = None):
        pass

    @abc.abstractmethod
    def fit_by_cross_validation(self, X, y, number_of_splits: int = 5, parameter_grid: dict = None, label_names: list = None):
        pass

    @abc.abstractmethod
    def store(self, dataset, path):
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
    def predict_proba(self, X, labels):
        pass
