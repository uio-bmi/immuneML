import abc


class EncodingParameterParser(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def parse(params: dict):
        pass
