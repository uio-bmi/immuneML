from dataclasses import dataclass
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.ClusteringSetting import ClusteringSetting


@dataclass
class ClusteringState:
    name: str
    dataset: Dataset
    metrics: List[str]
    clustering_settings: List[ClusteringSetting]
    result_path: Path = None
    label_config: LabelConfiguration = None
    report_results: List[ReportResult] = None


class ClusteringInstruction(Instruction):
    """
    Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
    clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. Finally, it
    provides options to include a set of reports to visualize the results.

    Specification arguments:

    - dataset (str): name of the dataset to be clustered

    - metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
      metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
      clusters can be compared against a list of predefined labels)

    - labels (list): an optional list of labels to use for external evaluation of clustering

    - clustering_settings (list): a list of combinations of encoding, optional dimensionality reduction algorithm, and
      the clustering algorithm that will be evaluated

    - reports (list): a list of reports to be run on the clustering results or the algorithms

    YAML specification:

    .. indent with spaces
    .. code-block: yaml

        my_clustering_instruction:
            type: Clustering
            dataset: d1
            metrics: [adjusted_rand_score, adjusted_mutual_info_score]
            labels: [epitope, v_call]
            clustering_settings:
                - encoding: e1
                  dim_reduction: pca
                  method: k_means1
                - encoding: e2
                  method: dbscan
            reports: [rep1, rep2]

    """

    def __init__(self, dataset: Dataset, metrics: List[str], clustering_settings: List[ClusteringSetting],
                 name: str, label_config: LabelConfiguration = None, reports: List[Report] = None):
        self.state = ClusteringState(name, dataset, metrics, clustering_settings, label_config=label_config)
        self.reports = reports

    def run(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path)
        print("In the clustering instruction!")
        return self.state
