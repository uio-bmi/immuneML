from immuneML.reports.Report import Report


class EncodingReport(Report):
    """
    Encoding reports show some type of features or statistics about an encoded dataset, or may in some cases
    export relevant sequences or tables.

    When running the :ref:`TrainMLModel` instruction, encoding reports can be specified inside the 'selection' or 'assessment' specification under the key 'reports:encoding'.

    Alternatively, when running the :ref:`ExploratoryAnalysis` instruction, encoding reports can be specified under 'reports'.
    """

    @staticmethod
    def get_title():
        return "Encoding reports"
