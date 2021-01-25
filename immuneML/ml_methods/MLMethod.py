import abc
from pathlib import Path

from immuneML.data_model.encoded_data.EncodedData import EncodedData


class MLMethod(metaclass=abc.ABCMeta):
    """
    Base class for different machine learning methods, defining which functions should be implemented. These public functions are the only ones that
    will be used outside the method, during training, assessment or while making predictions. Most often the methods will be classifiers (binary or
    multi-class) that should learn some label on either immune repertoires (sets of receptor sequences), receptors (paired sequences) or receptor
    sequences (lists of amino acids).

    Here we refer to machine learning methods (algorithms) as a method that, given a set of examples and corresponding labels, constructs a model
    (such as logistic regression), whereas we define the model to be already fit to data using the learning method (algorithm), such as logistic
    regression with specific coefficients.

    The functions of this class provide a standard set of ML functions: fitting the model (with or without cross-validation) and making predictions
    (either class predictions or class probabilities if possible). Other functions provide for various utilities, such as storing and loading the
    model, checking if it was fit already, retrieving coefficients for user-friendly output etc.

    Note that when providing class probabilities the classes should have a specific (constant) order, and in case of binary classification, they
    should be ordered so that the negative class comes first and the positive one comes second. For this handling classes, see
    py:`immuneML.ml_methods.util.Util.Util.make_binary_class_mapping` method that will automatically create class mapping for binary classification.

    """
    def __init__(self):
        self.ml_details_path = None
        self.predictions_path = {}
        self.name = None

    @abc.abstractmethod
    def fit(self, encoded_data: EncodedData, label_name: str, cores_for_training: int = 2):
        """
        The fit function fits the parameters of the machine learning model.

        Arguments:

            encoded_data (EncodedData): an instance of EncodedData class which includes encoded examples (repertoires, receptors or sequences), their
                labels, names of the features and other additional information. Most often, only examples and labels will be used. Examples are either a
                dense numpy matrix or a sparse matrix, where columns correspond to features and rows correspond to examples. There are a few encodings
                which make multidimensional outputs that do not follow this pattern, but they are tailored to specific ML methods which require such input
                (for instance, one hot encoding and ReceptorCNN method).

            label_name (str): name of the label for which the classifier will be created. immuneML also supports multi-label classification, but it is
                handled outside MLMethod class by creating an MLMethod instance for each label. This means that each MLMethod should handle only one label.

            cores_for_training (int): if parallelization is available in the MLMethod (and the availability depends on the specific classifier), this
                is the number of processes that will be creating when fitting the model to speed up the computation.

        Returns:

            it doesn't return anything, but fits the model parameters instead

        """
        pass

    @abc.abstractmethod
    def predict(self, encoded_data: EncodedData, label_name: str):
        """
        The predict function predicts the class for the given label across examples provided in encoded data.

        Arguments:

            encoded_data (EncodedData): an instance of EncodedData class which includes encoded examples (repertoires, receptors or sequences), their
                labels, names of the features and other additional information. Most often, only examples and labels will be used. Examples are either a
                dense numpy matrix or a sparse matrix, where columns correspond to features and rows correspond to examples. There are a few encodings
                which make multidimensional outputs that do not follow this pattern, but they are tailored to specific ML methods which require such input
                (for instance, one hot encoding and ReceptorCNN method).

            label_name (str): name of the label for which the classifier will be created. immuneML also supports multi-label classification, but it is
                handled outside MLMethod class by creating an MLMethod instance for each label. This means that each MLMethod should handle only one label.

        Returns:

            a dictionary where the key is the label_name and the value is a list of class predictions (one prediction per example):
            e.g., {label_name: [class1, class2, class2, class1]}

        """
        pass

    @abc.abstractmethod
    def fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int = 5, label_name: str = None, cores_for_training: int = -1,
                                optimization_metric=None):
        """
        The fit_by_cross_validation function should implement finding the best model hyperparameters through cross-validation. In immuneML,
        preprocessing, encoding and ML hyperparameters can be optimized by using nested cross-validation (see TrainMLModelInstruction for more
        details). This function is in that setting the third level of nested cross-validation as it can optimize only over the model hyperparameters.
        It represents an alternative to optimizing the model hyperparameters in the TrainMLModelInstruction. Which one should be used depends on the
        use-case and specific models: models based on scikit-learn implementations come with this option by default (see SklearnMethod class), while
        custom classifiers typically do not implement this and just call fit() function and throw a warning instead.

        Arguments:

            encoded_data (EncodedData): an instance of EncodedData class which includes encoded examples (repertoires, receptors or sequences), their
                labels, names of the features and other additional information. Most often, only examples and labels will be used. Examples are either a
                dense numpy matrix or a sparse matrix, where columns correspond to features and rows correspond to examples. There are a few encodings
                which make multidimensional outputs that do not follow this pattern, but they are tailored to specific ML methods which require such input
                (for instance, one hot encoding and ReceptorCNN method).

            number_of_splits (int): number of splits for the cross-validation to be performed for selection the best hyperparameters of the ML model;
                note that if this is used in combination with nested cross-validation in TrainMLModel instruction, it can result in very few examples in
                each split depending on the orginal dataset size and the nested cross-validation setup.

            label_name (str): name of the label for which the classifier will be created. immuneML also supports multi-label classification, but it is
                handled outside MLMethod class by creating an MLMethod instance for each label. This means that each MLMethod should handle only one label.

            cores_for_training (int): number of processes to be used during the cross-validation for model selection

            optimization_metric (str): the name of the optimization metric to be used to select the best model during cross-validation; when used with
                TrainMLModel instruction which is almost exclusively the case when the immuneML is run from the specification, this maps to the
                optimization metric in the instruction.

        Returns:

            it doesn't return anything, but fits the model parameters instead

        """
        pass

    @abc.abstractmethod
    def store(self, path: Path, feature_names: list = None, details_path: Path = None):
        """
        The store function stores the object on which it is called so that it can be imported later using load function. It typically uses pickle,
        yaml or similar modules to store the information. It can store one or multiple files.

        Arguments:

            path (Path): path to folder where to store the model

            feature_names (list): list of feature names in the encoded data; this can be stored as well to make it easier to map linear models to
                specific features as provided by the encoded (e.g., in case of logistic regression, this feature list defines what coefficients refer to)

            details_path (Path): path to folder where to store the details of the model. The details can be there to better understand the model but
                are not mandatory and are typically not loaded with the model afterwards. This is user-friendly file that can be examined manually by the
                user. It does not have to be created or can be created at the same folder as the path parameters points to. In practice, when used with
                TrainMLModel instruction, this parameter will either be None or have the same value as path parameter.

        Returns:

            it does not have a return value

        """
        pass

    @abc.abstractmethod
    def load(self, path: Path):
        """
        The load function can load the model given the folder where the same class of the model was previously stored using the store function.
        It reads in the parameters of the model and sets the values to the object attributes so that the model can be reused. For instance, this is
        used in MLApplication instruction when the previously trained model is applied on a new dataset.

        Arguments:

            path (Path): path to the folder where the model was stored using store() function

        Returns:

            it does not have a return value, but sets the attribute values of the object instead

        """
        pass

    @abc.abstractmethod
    def get_model(self, label_name: str) -> dict:
        """
        The get_model function returns the parameters of the model. This is usually used to show the parameters of the model in a user-friendly way.

        Arguments:

            label_name (str): the name of the label for which the model was trained. It is useful to check if the label_name and the model label
            match.

        Returns:

            a dictionary with model parameter values; the values could be of any complexity (e.g., dictionaries, lists)

        """
        pass

    @abc.abstractmethod
    def check_if_exists(self, path: Path) -> bool:
        """
        The check_if_exists function checks if there is a stored model on the given path. Might be useful in the future for implementing checkpoints.
        See SklearnMethod for example usage.

        Arguments:

            path (Path): path to folder where it should be checked if the model was stored previously

        Returns:

            True/False: whether the model was stored previously on the given path or not

        """
        pass

    @abc.abstractmethod
    def get_classes_for_label(self, label_name: str):
        """The get_classes_for_label function returns a list of classes for the given label_name."""
        pass

    @abc.abstractmethod
    def get_params(self, label_name: str):
        """
        Returns the parameters of the model, similar to get_model function, except it does not include any additional info other than model
        coefficients for the given label_name.
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, encoded_data: EncodedData, label_name: str):
        """
        The predict_proba function predicts class probabilities for the given label if the model supports probabilistic output. If not, it should
        raise a warning and return predicted classes without probabilities.

        Note that when providing class probabilities the classes should have a specific (constant) order, and in case of binary classification, they
        should be ordered so that the negative class comes first and the positive one comes second. For this handling classes, see
        py:`immuneML.ml_methods.util.Util.Util.make_binary_class_mapping` method that will automatically create class mapping for binary classification.

        Arguments:

            encoded_data (EncodedData): an object of EncodedData class where the examples attribute should be used to make predictions. examples
            attribute includes encoded examples in matrix format (numpy 2D array or a sparse matrix depending on the encoding). EncodedData object
            provided here can include labels (in model assessment scenario) or not (when predicting the class probabilities on new data which has not
            been labels), so the labels attribute of the EncodedData object should NOT be used in this function, even if it is set.

            label_name (str): the name of the label for which the prediction should be made. It can be used to check if it matches the label that the
            model has been trained for and if not, an exception should be thrown. It is often an AssertionError as this can be checked before any
            prediction is made, but could also be a RuntimeError. It both cases, it should include a user-friendly message.

        Returns:

            a dictionary where the key is the label name and the value a 2D numpy array with class probabilities of dimension
            [number_of_examples x number_of_classes_for_label], for instance for label CMV where the class can be either True or False and there are
            3 examples to predict the class probabilities for:
            {CMV: [[0.2, 0.8], [0.55, 0.45], [0.98, 0.02]]}

        """
        pass

    @abc.abstractmethod
    def get_label(self) -> str:
        """Returns the name of the label for which the model was fitted."""
        pass

    @abc.abstractmethod
    def get_package_info(self) -> str:
        """
        Returns the package and version used for implementing the ML method if an external package was used or immuneML version if it is custom
        implementation. See py:`immuneML.ml_methods.SklearnMethod.SklearnMethod` and py:`immuneML.ml_methods.ProbabilisticBinaryClassifier.ProbabilisticBinaryClassifier`
         for examples of both.
        """
        pass

    @abc.abstractmethod
    def get_feature_names(self) -> list:
        """
        Returns the list of feature names (a list of strings) if available where the feature names were provided by the encoder in the
        EncodedData object.
        """
        pass

    @abc.abstractmethod
    def can_predict_proba(self) -> bool:
        """
        Returns whether the ML model can be used to predict class probabilities or class assignment only.
        """
        return False
