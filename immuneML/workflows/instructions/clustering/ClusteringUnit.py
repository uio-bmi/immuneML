from dataclasses import dataclass

from immuneML.ml_methods.Clustering import Clustering
from immuneML.ml_methods import DimensionalityReduction
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult


@dataclass
class ClusteringUnit:
    dataset: Dataset
    report: Report
    clustering_method: Clustering
    dimensionality_reduction: DimensionalityReduction = None
    encoder: DatasetEncoder = None
    report_result: ReportResult = None
    label_config: LabelConfiguration = None
    number_of_processes: int = 1
