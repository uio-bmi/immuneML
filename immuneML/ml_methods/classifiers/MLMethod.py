import abc
from pathlib import Path
import logging

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment import Label
from immuneML.ml_methods.util.Util import Util


class MLMethod(metaclass=abc.ABCMeta):
    """
    ML method classifiers are algorithms which can be trained to predict some label on immune
    repertoires, receptors or sequences.

    These methods can be trained using the :ref:`TrainMLModel` instruction, and previously trained
    models can be applied to new data using the :ref:`MLApplication` instruction.

    When choosing which ML method(s) are most suitable for your use-case, please consider the following table:

    .. csv-table:: ML methods properties
       :file: ../../source/_static/files/ml_methods_properties.csv
       :header-rows: 1
    """
    DOCS_TITLE = "Classifiers"

    def __init__(self):
        self.name = None
        self.label = None
        self.class_mapping = None
        self.feature_names = None
        self.optimization_metric = None
        self.result_path = None

    def fit(self, encoded_data: EncodedData, label: Label, optimization_metric: str = None, cores_for_training: int = 2):
        """
        The fit method is called by the MLMethodTrainer to fit a model to a specific encoded dataset.
        This method internally calls methods to initialise and perform model fitting.

        This method should not be overwritten.
        """
        self._initialize_fit(encoded_data, label, optimization_metric)
        self._fit(encoded_data, cores_for_training)

    def _initialize_fit(self, encoded_data: EncodedData, label: Label, optimization_metric: str):
        """
        Sets parameters and performs checks prior to fitting a model to encoded data.
        The following parameters will be set and are available to be used inside '_fit':

        - label (Label): the label for which the classifier will be created.
          Note that an individual MLMethod should handle only one label (individual methods will be trained for each provided label in TrainMLModelInstruction).

        - class_mapping (dict): This class mapping may be used to convert the original classes (strings, numbers or booleans) to numeric values.
          This can be useful for example if an internally used model only accepts numeric classes.
          It is not always necessary to use class_mapping: positive and negative classes may be directly retrieved from the Label object.

        - optimization_metric (str): The name of the optimization metric which is used to select the best model during cross-validation.
          This metric may be used internally for example for hyperparameter selection or earlystopping.

        - feature_names (list): The names of the features from the encoded data, which are exported alongside the model for transparency.

        The method furthermore checks and shows a warning if example weights are supplied for a model which cannot use
        example weights during fitting.

        This method should not be overwritten.
        """
        self.label = label
        self.class_mapping = Util.make_class_mapping(encoded_data.labels[label.name], label.positive_class)
        self.optimization_metric = optimization_metric
        self.feature_names = encoded_data.feature_names

        if encoded_data.example_weights is not None and not self.can_fit_with_example_weights():
            logging.warning(f"{self.__class__.__name__}: cannot fit this classifier with example weights, fitting without example weights instead... Example weights will still be applied when computing evaluation metrics after fitting.")

    @abc.abstractmethod
    def _fit(self, encoded_data: EncodedData, cores_for_training: int = 2):
        """
        Fits the parameters of the machine learning model. This method should be implemented when adding a new MLMethod.
        Note that '_initialize_fit' is called before '_fit', such that any parameters set during initialization are
        available during fitting.

        Arguments:

            encoded_data (EncodedData): an instance of EncodedData class which includes encoded examples (repertoires, receptors or sequences), their
                labels, names of the features and other additional information. Most often, only examples and labels will be used. Examples are either a
                dense numpy matrix or a sparse matrix, where columns correspond to features and rows correspond to examples. There are a few encodings
                which make multidimensional outputs that do not follow this pattern, but they are tailored to specific ML methods which require such input
                (for instance, one hot encoding and ReceptorCNN method).

            cores_for_training (int): number of processes to be used for optional parallelization.

        Returns:

            it doesn't return anything, but fits the model parameters instead
        """
        pass

    def fit_by_cross_validation(self, encoded_data: EncodedData, label: Label, optimization_metric, number_of_splits: int = 5, cores_for_training: int = 2):
        """
        See also: _fit_by_cross_validation.

        This method should not be overwritten.
        """
        self._initialize_fit(encoded_data, label, optimization_metric)
        self._fit_by_cross_validation(encoded_data, number_of_splits, cores_for_training)

    def _fit_by_cross_validation(self, encoded_data: EncodedData, number_of_splits: int, cores_for_training: int):
        """
        Like _fit, but this method first models through an additional (third) level of cross-validation.
        For most ML methods it is typically not defined. An exception is scikit-learn methods,
        see SklearnMethod for a more detailed example implementation.

        Detailed info:
        In immuneML, preprocessing, encoding and ML hyperparameters can be optimized by using nested cross-validation (see TrainMLModelInstruction for more
        details). This function is in that setting the third level of nested cross-validation which can optimize only over the model hyperparameters.
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

            cores_for_training (int): number of processes to be used for optional parallelization.


        Returns:

            it doesn't return anything, but fits the model parameters instead

        """
        logging.warning(f"{self.__class__.__name__}: fitting by cross-validation is not implemented for this classifier: fitting one model instead...")
        self._fit(encoded_data=encoded_data, cores_for_training=cores_for_training)

    def _assert_matching_label(self, label: Label):
        """Ensures the label for prediction is matching. This method should not be overwritten."""
        assert self.label == label, f"{self.__class__.__name__}: this method is fitted for {self.label}, not {label}"

    def predict(self, encoded_data: EncodedData, label: Label):
        """Safely calls '_predict' after checking the label is matching. This method should not be overwritten."""
        self._assert_matching_label(label)
        return self._predict(encoded_data)

    def predict_proba(self, encoded_data: EncodedData, label: Label):
        """Safely calls '_predict_proba' after checking the label is matching. This method should not be overwritten."""
        self._assert_matching_label(label)
        return self._predict_proba(encoded_data)

    @abc.abstractmethod
    def _predict(self, encoded_data: EncodedData):
        """
        The predict function predicts the class for the given label across examples provided in encoded data.

        Arguments:

            encoded_data (EncodedData): an instance of EncodedData class which includes encoded examples (repertoires, receptors or sequences), their
                labels, names of the features and other additional information. Most often, only examples and labels will be used. Examples are either a
                dense numpy matrix or a sparse matrix, where columns correspond to features and rows correspond to examples. There are a few encodings
                which make multidimensional outputs that do not follow this pattern, but they are tailored to specific ML methods which require such input
                (for instance, one hot encoding and ReceptorCNN method).

        Returns:

            a dictionary where the key is the label_name and the value is a list of class predictions (one prediction per example):
            e.g., {label_name: [class1, class2, class2, class1]}

        """
        pass

    @abc.abstractmethod
    def _predict_proba(self, encoded_data: EncodedData):
        """
        The predict_proba function predicts class probabilities for the given label if the model supports probabilistic output.
        If not, it should raise a warning and return predicted classes without probabilities.

        The function will return a nested dictionary. The key(s) of the outer dictionary represent the label name(s),
        and the keys of the inner dictionary the class names of the respective label.
        The utility function py:mod:`immuneML.ml_methods.util.Util.Util.make_binary_class_mapping` may be used to
        handle mapping of class names to an internal representation for binary classification.

        Arguments:

            encoded_data (EncodedData): an object of EncodedData class where the examples attribute should be used to make predictions. examples
            attribute includes encoded examples in matrix format (numpy 2D array or a sparse matrix depending on the encoding). EncodedData object
            provided here can include labels (in model assessment scenario) or not (when predicting the class probabilities on new data which has not
            been labels), so the labels attribute of the EncodedData object should NOT be used in this function, even if it is set.

        Returns:

            a nested dictionary where the outer keys represent label names, inner keys represent class names for the respective
            label, and innermost values are 1D numpy arrays with class probabilities.
            For example for instance for label CMV where the class can be either True or False and there are
            3 examples to predict the class probabilities for: {CMV: {True: [0.2, 0.55, 0.98], False: [0.8, 0.45, 0.02]}}

        """
        pass

    @abc.abstractmethod
    def store(self, path: Path):
        """
        The store function stores the object on which it is called so that it can be imported later using load function. It typically uses pickle,
        yaml or similar modules to store the information. It can store one or multiple files.

        Arguments:

            path (Path): path to folder where to store the model

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
    def get_params(self) -> dict:
        """
        Returns the model parameters and their values in a readable yaml-friendly way (a dictionary consisting of ints, floats, strings, lists and dictionaries).
        This may simply be vars(self), but if an internal (sklearn) model is fitted, the parameters of the internal model should
        be included as well.
        """
        pass

    @abc.abstractmethod
    def can_predict_proba(self) -> bool:
        """
        Returns whether the ML model can be used to predict class probabilities or class assignment only.
        This method should be overwritten to return True if probabilities can be predicted, and False if they cannot.
        """
        pass

    @abc.abstractmethod
    def can_fit_with_example_weights(self) -> bool:
        """
        Returns a boolean value indicating whether the model can be fit with example weights.
        Example weights allow to up-weight the importance of certain examples, and down-weight the importance of others.
        """
        pass

    @abc.abstractmethod
    def get_compatible_encoders(self):
        """Returns a list of compatible encoders. This method should be overwritten for every MLMethod.

        for example:

        from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder

        return [EvennessProfileEncoder]
        """
        pass

    def check_encoder_compatibility(self, encoder):
        """
        Checks whether the given encoder is compatible with this ML method, and throws an error if it is not.

        This method should not be overwritten.
        """
        is_valid = False

        for encoder_class in self.get_compatible_encoders():
            if issubclass(encoder.__class__, encoder_class):
                is_valid = True
                break

        if not is_valid:
            raise ValueError(f"{encoder.__class__.__name__} is not compatible with ML Method {self.__class__.__name__}. "
                             f"Please use one of the following encoders instead: {', '.join([enc_class.__name__ for enc_class in self.get_compatible_encoders()])}")

    def get_package_info(self) -> str:
        """
        Should return the relevant version numbers of immuneML and all external packages that were used for the immuneML implementation.
        This information will be exported and is required for transparency and reproducibility.

        This method should be overwritten to add any additional packages if necessary. For instance, versions of scikit-learn.
        """
        return Util.get_immuneML_version()

    def get_feature_names(self) -> list:
        return self.feature_names

    def get_label_name(self) -> str:
        return self.label.name

    def get_classes(self):
        return self.label.values

    def get_positive_class(self):
        return self.label.positive_class
