import abc


class MLMethod(metaclass=abc.ABCMeta):
    """
        Base class for wrappers for different ML/DL methods
        adapted to work with Dataset objects and use their encoded_data
        attribute to access the data;

        These classes (MLMethod and subclasses) assume that encoded_data has
        the following format: {"repertoires" : scipy.sparse.csc_matrix, "labels": np.ndarray}

        Other parameters in addition to repertoires and labels can be included,
        such as label_names, features names etc.
    """

    @abc.abstractmethod
    def fit(self, X, y, label_names: list = None):
        pass

    @abc.abstractmethod
    def predict(self, X, label_names: list = None):
        pass

    @abc.abstractmethod
    def fit_by_cross_validation(self, X, y, number_of_splits: int = 5, parameter_grid: dict = None, label_names: list = None):
        pass

    @abc.abstractmethod
    def store(self, path):
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
