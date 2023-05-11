
import abc
from pathlib import Path

from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.environment.EnvironmentSettings import EnvironmentSettings


class GenerativeModel(UnsupervisedMLMethod):
    """

    Base class for ML methods performing generative modeling. The classes inheriting GenerativeModel have to implement:
        - the __init__() method,
        - get_params(label),
        - _get_ml_model(), and
        - generate()
    Other methods can also be overwritten if needed.
    The arguments and specification described bellow applied for all classes inheriting GenerativeModel.

    Arguments:

        parameters: a dictionary of parameters used for the various generators. Must contain the keys "amount" and
        "sequence_type".

    YAML specification:

        ml_methods:
            gen_model:
                LSTM: # name of the class inheriting GenerativeModel
                    # parameterization
                    batch_size: 64
                    rnn_units: 128
                    embedding_dim: 256
                    epochs: 5
                    amount: 200
            PWM_without_params: PWM # name of another class inheriting GenerativeModel, but without parameters, relying
            on default parameters.

    """

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(GenerativeModel, self).__init__()

        if parameters is not None and "show_warnings" in parameters:
            self.show_warnings = parameters.pop("show_warnings")
        else:
            self.show_warnings = True

        self._parameter_grid = parameter_grid
        self._parameters = parameters
        self._amount = parameters["amount"]
        self.model = None
        self.alphabet = EnvironmentSettings.get_sequence_alphabet(SequenceType(parameters["sequence_type"]))
        self.char2idx = {u: i for i, u in enumerate(self.alphabet)}

    def fit(self, encoded_data, cores_for_training: int = 2):
        self.model = self._fit(encoded_data.examples, cores_for_training)

    def generate(self):
        pass

    @abc.abstractmethod
    def _fit(self, X, cores_for_training: int = 1):
        pass

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        pass

    def _get_model_filename(self):
        return FilenameHandler.get_filename(self.__class__.__name__, "")

    def load(self, path: Path, details_path: Path = None):
        pass

    def check_if_exists(self, path: Path):
        pass

    @abc.abstractmethod
    def _get_ml_model(self):
        pass

    @abc.abstractmethod
    def get_params(self):
        '''Returns the model parameters in a readable yaml-friendly way (consisting of lists, dictionaries and strings).'''
        return self._parameters

    def get_package_info(self) -> str:
        pass
    def get_feature_names(self) -> list:
        pass
    @abc.abstractmethod
    def get_compatible_encoders(self):
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.char_to_int.CharToIntEncoder import CharToIntEncoder

        return [CharToIntEncoder, OneHotEncoder]

    @staticmethod
    def get_usage_documentation(model_name):
        return f"""
        
        TODO
        
        
        """