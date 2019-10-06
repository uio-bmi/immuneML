# quality: gold

import abc

from source.encodings.EncoderParams import EncoderParams


class DatasetEncoder(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def create_encoder(dataset, params: dict = None):
        pass

    @abc.abstractmethod
    def encode(self, dataset, params: EncoderParams):
        pass

    def set_context(self, context: dict):
        return self

    @abc.abstractmethod
    def store(self, encoded_dataset, params: EncoderParams):
        pass
