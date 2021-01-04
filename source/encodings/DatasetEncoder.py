import abc
import os
import pickle
import shutil
from typing import List
from pathlib import Path

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.encodings.EncoderParams import EncoderParams


class DatasetEncoder(metaclass=abc.ABCMeta):
    """

    YAML specification:

        encodings:
            e1: <encoder_class> # encoding without parameters

            e2:
                <encoder_class>: # encoding with parameters
                    parameter: value
    """

    @staticmethod
    @abc.abstractmethod
    def build_object(dataset, **params):
        pass

    @abc.abstractmethod
    def encode(self, dataset, params: EncoderParams):
        pass

    @staticmethod
    def load_encoder(encoder_file: Path):
        with encoder_file.open("rb") as file:
            encoder = pickle.load(file)
        return encoder

    @staticmethod
    def load_attribute(encoder, encoder_file: Path, attribute: str):
        if encoder_file is not None and str(encoder_file) != '':
            assert False, "need to update this with the paths"
            setattr(encoder, attribute, f"{os.path.dirname(encoder_file)}/{os.path.basename(getattr(encoder, attribute))}")
            assert os.path.isfile(getattr(encoder, attribute)), f"{type(encoder).__name__}: could not load {attribute} from {getattr(encoder, attribute)}."
        return encoder

    @staticmethod
    def store_encoder(encoder, encoder_file: Path):
        with encoder_file.open("wb") as file:
            pickle.dump(encoder, file)

        encoder_dir = encoder_file.parent
        for file in encoder.get_additional_files():
            shutil.copy(file, encoder_dir / file.name)

        return encoder_file

    @staticmethod
    def get_additional_files() -> List[str]:
        return []

    def set_context(self, context: dict):
        return self

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params.result_path)
