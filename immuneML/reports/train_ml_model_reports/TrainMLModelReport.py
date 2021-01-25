from immuneML.reports.Report import Report


class TrainMLModelReport(Report):
    """
    Train ML model reports plot general statistics or export data of multiple models simultaneously when running the :ref:`TrainMLModel` instruction.

    In the :ref:`TrainMLModel` instruction, train ML model reports can be specified under 'reports'.
    """

    @staticmethod
    def get_title():
        return "Train ML model reports"
