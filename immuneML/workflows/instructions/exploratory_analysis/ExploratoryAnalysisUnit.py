from dataclasses import dataclass

import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult


@dataclass
class ExploratoryAnalysisUnit:
    dataset: Dataset
    report: Report
    preprocessing_sequence: list = None
    encoder: DatasetEncoder = None
    label_config: LabelConfiguration = None
    number_of_processes: int = 1
    report_result: ReportResult = None
    dim_reduction: str = None
    dim_reduction_result_matrix: np.ndarray = None
