from abc import ABC

from source.reports.Report import Report


class HyperparameterReport(Report, ABC):
    """
    Hyperparameter reports plot general statistics of multiple models simultaneously when running the :ref:`TrainMLModel` instruction.

    In the :ref:`TrainMLModel` instruction, hyperparameter reports can be specified under 'assessment/reports/hyperparameter'.
    """

    @staticmethod
    def get_title():
        return "Hyperparameter reports"
