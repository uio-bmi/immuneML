from pathlib import Path

from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.Report import Report


class TrainMLModelReport(Report):
    """
    Train ML model reports plot general statistics or export data of multiple models simultaneously when running the :ref:`TrainMLModel` instruction.

    In the :ref:`TrainMLModel` instruction, train ML model reports can be specified under 'reports'.
    """

    def __init__(self, name: str = None, state: TrainMLModelState = None, result_path: Path = None):
        super().__init__(name)
        self.state = state
        self.result_path = result_path

    @staticmethod
    def get_title():
        return "Train ML model reports"
