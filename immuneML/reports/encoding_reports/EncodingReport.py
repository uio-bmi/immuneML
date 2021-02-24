from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.reports.Report import Report


@dataclass
class EncodingReport(Report):
    """
    Encoding reports show some type of features or statistics about an encoded dataset, or may in some cases
    export relevant sequences or tables.

    When running the :ref:`TrainMLModel` instruction, encoding reports can be specified inside the 'selection' or 'assessment' specification under the key 'reports:encoding'.
    Alternatively, when running the :ref:`ExploratoryAnalysis` instruction, encoding reports can be specified under 'reports'.

    When using the reports with instructions such as ExploratoryAnalysis or TrainMLModel, the arguments defined below are set at runtime by the instruction.
    Concrete classes inheriting EncodingReport may include additional parameters that will be set by the user in the form of input arguments.

    Arguments:

        dataset (Dataset): an encoded dataset where encoded_data attribute is set to an instance of EncodedData object
        result_path (Path): path where the results will be stored (plots, tables, etc.)
        name (str): user-defined name of the report that will be shown in the HTML overview later

    """

    dataset: Dataset = None
    result_path: Path = None
    name: str = None

    @staticmethod
    def get_title():
        return "Encoding reports"
