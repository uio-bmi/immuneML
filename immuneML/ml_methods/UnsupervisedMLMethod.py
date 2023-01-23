import abc
from pathlib import Path


class UnsupervisedMLMethod(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = None

    @abc.abstractmethod
    def fit(self, encoded_data, cores_for_training: int = 2):
        pass

    @abc.abstractmethod
    def store(self, path: Path, feature_names: list = None, details_path: Path = None):
        pass

    @abc.abstractmethod
    def load(self, path: Path):
        pass

    @abc.abstractmethod
    def check_if_exists(self, path: Path) -> bool:
        pass

    @abc.abstractmethod
    def get_params(self):
        pass

    @abc.abstractmethod
    def get_package_info(self) -> str:
        pass

    @abc.abstractmethod
    def get_feature_names(self) -> list:
        pass

    @abc.abstractmethod
    def get_compatible_encoders(self):
        pass

    def check_encoder_compatibility(self, encoder):
        is_valid = False

        for encoder_class in self.get_compatible_encoders():
            if issubclass(encoder.__class__, encoder_class):
                is_valid = True
                break

        if not is_valid:
            raise ValueError(f"{encoder.__class__.__name__} is not compatible with ML Method {self.__class__.__name__}. "
                             f"Please use one of the following encoders instead: {', '.join([enc_class.__name__ for enc_class in self.get_compatible_encoders()])}")

