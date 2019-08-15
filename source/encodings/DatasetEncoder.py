# quality: gold

import abc

from source.encodings.EncoderParams import EncoderParams


class DatasetEncoder(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def create_encoder(dataset):
        pass

    @abc.abstractmethod
    def encode(self, dataset, params: EncoderParams):
        pass

    @abc.abstractmethod
    def store(self, encoded_dataset, params: EncoderParams):
        pass
