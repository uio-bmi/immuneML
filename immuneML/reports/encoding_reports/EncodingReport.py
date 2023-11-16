from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.Report import Report


class EncodingReport(Report):
    """
    Encoding reports show some type of features or statistics about an encoded dataset, or may in some cases
    export relevant sequences or tables.

    When running the :ref:`TrainMLModel` instruction, encoding reports can be specified inside the 'selection' or 'assessment' specification under the key 'reports/encoding'.
    Example:

    .. indent with spaces
    .. code-block:: yaml

        my_instruction:
            type: TrainMLModel
            selection:
                reports:
                    encoding:
                        - my_encoding_report
                # other parameters...
            assessment:
                reports:
                    encoding:
                        - my_encoding_report
                # other parameters...
            # other parameters...

    Alternatively, when running the :ref:`ExploratoryAnalysis` instruction, encoding reports can be specified under 'report'. Example:

    .. indent with spaces
    .. code-block:: yaml

        my_instruction:
            type: ExploratoryAnalysis
            analyses:
                my_first_analysis:
                    report: my_encoding_report
                    # other parameters...
            # other parameters...

    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        """
        The arguments defined below are set at runtime by the instruction.
        Concrete classes inheriting EncodingReport may include additional parameters that will be set by the user in the form of input arguments.

        dataset (Dataset): an encoded dataset where encoded_data attribute is set to an instance of EncodedData object
        result_path (Path): path where the results will be stored (plots, tables, etc.)
        name (str): user-defined name of the report that will be shown in the HTML overview later
        number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.
        """
        super().__init__(name, number_of_processes)
        self.dataset = dataset
        self.result_path = result_path

    @staticmethod
    def get_title():
        return "Encoding reports"
