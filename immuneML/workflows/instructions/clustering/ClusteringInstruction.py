import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import sklearn

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_metrics.ClusteringMetric import is_internal, is_external
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting, ClusteringItem
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


@dataclass
class ClusteringState:
    name: str
    dataset: Dataset
    metrics: List[str]
    clustering_settings: List[ClusteringSetting]
    clustering_items: List[ClusteringItem] = field(default_factory=list)
    result_path: Path = None
    label_config: LabelConfiguration = None
    predictions_path: Path = None
    cl_item_report_results: Dict[str, Dict[str, List[ReportResult]]] = None
    clustering_report_results: List[ReportResult] = field(default_factory=list)


class ClusteringInstruction(Instruction):
    """
    Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
    clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. Finally, it
    provides options to include a set of reports to visualize the results.

    .. note::

        This is an experimental feature in version 3.0.0a1.

    Specification arguments:

    - dataset (str): name of the dataset to be clustered

    - metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
      metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
      clusters can be compared against a list of predefined labels)

    - labels (list): an optional list of labels to use for external evaluation of clustering

    - clustering_settings (list): a list of combinations of encoding, optional dimensionality reduction algorithm, and
      the clustering algorithm that will be evaluated

    - reports (list): a list of reports to be run on the clustering results or the encoded data

    - number_of_processes (int): how many processes to use for parallelization

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

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
                 name: str, label_config: LabelConfiguration = None, reports: List[Report] = None,
                 number_of_processes: int = None):
        self.state = ClusteringState(name, dataset, metrics, clustering_settings, label_config=label_config)
        self.reports = reports
        self.number_of_processes = number_of_processes

    def run(self, result_path: Path):
        self._setup_paths(result_path)
        self._init_report_result_structure()

        predictions_df = self.state.dataset.get_metadata(self.state.label_config.get_labels_by_name(), return_df=True)
        predictions_df['example_id'] = self.state.dataset.get_example_ids()

        for cl_setting in self.state.clustering_settings:
            cl_item, predictions_df = self._run_clustering_setting(cl_setting, predictions_df)
            self.state.clustering_items.append(cl_item)

        predictions_df.to_csv(self.state.predictions_path, index=False)
        self._run_clustering_reports()

        return self.state

    def _setup_paths(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path)
        self.state.predictions_path = self.state.result_path / 'predictions.csv'

    def _run_clustering_setting(self, cl_setting: ClusteringSetting, predictions_df: pd.DataFrame) \
            -> Tuple[ClusteringItem, pd.DataFrame]:

        cl_setting_path = PathBuilder.build(self.state.result_path / cl_setting.get_key())

        dataset = DataEncoder.run(DataEncoderParams(dataset=self.state.dataset, encoder=cl_setting.encoder,
                                                    encoder_params=EncoderParams(model=cl_setting.encoder_params,
                                                                                 result_path=cl_setting_path,
                                                                                 pool_size=self.number_of_processes,
                                                                                 label_config=self.state.label_config,
                                                                                 learn_model=True,
                                                                                 encode_labels=False)))
        if cl_setting.dim_reduction_method is not None:
            dataset.encoded_data.dimensionality_reduced_data = cl_setting.dim_reduction_method.fit_transform(dataset)

        cl_setting.clustering_method.fit(dataset)
        predictions = cl_setting.clustering_method.predict(dataset)
        predictions_df[f'predictions_{cl_setting.get_key()}'] = predictions

        features = dataset.encoded_data.examples if cl_setting.dim_reduction_method is None \
            else dataset.encoded_data.dimensionality_reduced_data
        ext_performances, int_performances = self._evaluate_clustering(predictions_df, cl_setting.get_key(), features)

        cl_item = ClusteringItem(cl_setting=cl_setting, dataset=dataset, predictions=predictions,
                                 performance=None)

        # performances.to_csv(str(cl_setting_path / 'performances.csv'), index=False)
        self._run_reports(cl_item)

        cl_item.dataset.encoded_data = None

        return cl_item, predictions_df

    def _evaluate_clustering(self, predictions: pd.DataFrame, cl_setting_name: str, features) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.state.label_config is not None and self.state.label_config.get_label_count() > 0:

            external_performances = {label: {} for label in self.state.label_config.get_labels_by_name()}
            internal_performances = {}

            for metric in [m for m in self.state.metrics if is_internal(m)]:
                metric_fn = getattr(sklearn.metrics, metric)
                internal_performances[metric] = metric_fn(features, predictions[f'predictions_{cl_setting_name}'].values)

            for metric in [m for m in self.state.metrics if is_external(m)]:
                metric_fn = getattr(sklearn.metrics, metric)
                for label in self.state.label_config.get_labels_by_name():
                    external_performances[label][metric] = metric_fn(predictions[label].values,
                                                                     predictions[
                                                                         f'predictions_{cl_setting_name}'].values)

            return (pd.DataFrame(external_performances).reset_index().rename(columns={'index': 'metric'}),
                    pd.DataFrame(internal_performances))

    def _run_clustering_reports(self):
        report_path = PathBuilder.build(self.state.result_path / f'reports/')
        for report in self.reports:
            if isinstance(report, ClusteringReport):
                tmp_report = copy.deepcopy(report)
                tmp_report.result_path = report_path
                tmp_report.cl_state = self.state
                self.state.clustering_report_results.append(tmp_report.generate_report())

        if len(self.reports) > 0:
            gen_rep_count = len(self.state.clustering_report_results)
            print_log(f"{self.state.name}: generated {gen_rep_count} clustering reports.", True)

    def _run_reports(self, cl_item: ClusteringItem):
        report_path = PathBuilder.build(self.state.result_path / f'reports/{cl_item.cl_setting.get_key()}')
        for report in self.reports:
            if isinstance(report, EncodingReport):
                tmp_report = copy.deepcopy(report)
                tmp_report.result_path = report_path
                tmp_report.dataset = cl_item.dataset
                rep_result = tmp_report.generate_report()
                self.state.cl_item_report_results[cl_item.cl_setting.get_key()]['encoding'].append(rep_result)

        if len(self.reports) > 0:
            gen_rep_count = sum(len(reports) for rep_type, reports in
                                self.state.cl_item_report_results[cl_item.cl_setting.get_key()].items())
            print_log(f"{self.state.name}: generated {gen_rep_count} reports for setting "
                      f"{cl_item.cl_setting.get_key()}.", True)

    def _init_report_result_structure(self):
        self.state.cl_item_report_results = {
            cl_setting.get_key(): {'encoding': []} for cl_setting in self.state.clustering_settings
        }
