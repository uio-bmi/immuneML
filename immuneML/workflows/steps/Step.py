import abc

from immuneML.workflows.steps.StepParams import StepParams


class Step(metaclass=abc.ABCMeta):
    """
    This class encapsulates steps in the analysis which will likely be often used, such as:
        - dataset encoding
        - training of machine learning models
        - signal implanting in repertoires without any signals etc.

    For a custom analysis which is not likely to be repeated for different settings
    (e.g. such as with a different encoding), create a custom class inheriting
    AbstractProcess from workflows.processes package.
    """

    @staticmethod
    @abc.abstractmethod
    def run(input_params: StepParams = None):
        pass
