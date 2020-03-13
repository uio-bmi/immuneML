# quality: gold

import abc

from source.encodings.EncoderParams import EncoderParams


class DatasetEncoder(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def build_object(dataset, **params):
        pass

    @abc.abstractmethod
    def encode(self, dataset, params: EncoderParams):
        pass

    def set_context(self, context: dict):
        return self

    @abc.abstractmethod
    def store(self, encoded_dataset, params: EncoderParams):
        pass
