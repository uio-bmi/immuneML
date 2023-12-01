import abc

from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams


class ExampleWeightingStrategy(metaclass=abc.ABCMeta):

    def __init__(self, name):
        self.name = name

    @staticmethod
    @abc.abstractmethod
    def build_object(dataset, **params):
        pass

    @abc.abstractmethod
    def compute_weights(self, dataset, params: ExampleWeightingParams):
        pass

    def set_context(self, context: dict):
        return self