import abc


class PreprocessingParameterParser(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def parse(params: dict):
        pass
