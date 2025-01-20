import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import scipy.sparse
import sklearn
from scipy.sparse import issparse

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.ml_metrics.ClusteringMetric import is_internal, is_external
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting, ClusteringItem, \
    DataFrameWrapper
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


@dataclass
class ClusteringState:
    name: str
    dataset: Dataset
    metrics: List[str]
    split_config: SplitConfig
    clustering_settings: List[ClusteringSetting]
    clustering_items: Dict[str, Dict[str, ClusteringItem]] = field(default_factory=dict)
    sequence_type: SequenceType = SequenceType.AMINO_ACID
    region_type: RegionType = RegionType.IMGT_CDR3
    result_path: Path = None
    label_config: LabelConfiguration = None
    predictions_paths: Dict[str, Path] = None
    discovery_dataset: Dataset = None
    validation_dataset: Dataset = None
    cl_item_report_results: Dict[str, Dict[str, List[ReportResult]]] = None
    clustering_report_results: List[ReportResult] = field(default_factory=list)


class ClusteringInstruction(Instruction):
    """

    Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
    clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. The dataset is split
    into discovery and validation datasets and the clustering results are reported on both. Finally, it
    provides options to include a set of reports to visualize the results.

    For more details on choosing the clustering algorithm and its hyperparameters, see the paper:
    Ullmann, T., Hennig, C., & Boulesteix, A.-L. (2022). Validation of cluster analysis results on validation
    data: A systematic framework. WIREs Data Mining and Knowledge Discovery, 12(3), e1444.
    https://doi.org/10.1002/widm.1444


    **Specification arguments:**

    - dataset (str): name of the dataset to be clustered

    - metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
      metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
      clusters can be compared against a list of predefined labels)

    - labels (list): an optional list of labels to use for external evaluation of clustering

    - split_config (SplitConfig): how to perform splitting of the original dataset into discovery and validation data;
      for this parameter, specify: split_strategy (leave_one_out_stratification, manual, random), training percentage
      if split_strategy is random, and defaults of manual or leave one out stratification config for corresponding split
      strategy; all three options are illustrated here:

      .. indent with spaces
      .. code-block:: yaml

        split_config:
            split_strategy: manual
            manual_config:
                discovery_data: file_with_ids_of_examples_for_discovery_data.csv
                validation_data: file_with_ids_of_examples_for_validation_data.csv

      .. indent with spaces
      .. code-block:: yaml

        split_config:
            split_strategy: random
            training_percentage: 0.5

      .. indent with spaces
      .. code-block:: yaml

        split_config:
            split_strategy: leave_one_out_stratification
            leave_one_out_config:
                parameter: subject_id # any name of the parameter for split, must be present in the metadata
                min_count: 1 #  defines the minimum number of examples that can be present in the validation dataset.

    - clustering_settings (list): a list where each element represents a :py:obj:`~immuneML.workflows.clustering.clustering_run_model.ClusteringSetting`; a combinations of encoding,
      optional dimensionality reduction algorithm, and the clustering algorithm that will be evaluated

    - reports (list): a list of reports to be run on the clustering results or the encoded data

    - number_of_processes (int): how many processes to use for parallelization

    - sequence_type (str): whether to do analysis on the amino_acid or nucleotide level; this value is used only if
      nothing is specified on the encoder level

    - region_type (str): which part of the receptor sequence to analyze (e.g., IMGT_CDR3); this value is used only if
      nothing is specified on the encoder level

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        instructions:
            my_clustering_instruction:
                type: Clustering
                dataset: d1
                metrics: [adjusted_rand_score, adjusted_mutual_info_score]
                labels: [epitope, v_call]
                sequence_type: amino_acid
                region_type: imgt_cdr3
                split_config:
                    split_strategy: manual
                    manual_config:
                        discovery_data: file_with_ids_of_examples_for_discovery_data.csv
                        validation_data: file_with_ids_of_examples_for_validation_data.csv
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
                 number_of_processes: int = None, split_config: SplitConfig = None, sequence_type: SequenceType = None,
                 region_type: RegionType = None):
        self.state = ClusteringState(name, dataset, metrics, clustering_settings=clustering_settings,
                                     label_config=label_config, split_config=split_config, sequence_type=sequence_type,
                                     region_type=region_type)
        self.reports = reports
        self.number_of_processes = number_of_processes

    def run(self, result_path: Path):
        self._setup_paths(result_path)
        self._init_report_result_structure()

        self._split_dataset()

        self._run_on_discovery_data()
        self._run_on_validation_data()
        self._run_clustering_reports()

        return self.state

    def _run_on_discovery_data(self):
        self._run_on_cluster_settings(self.state.discovery_dataset, 'discovery')

    def _run_on_validation_data(self):
        self._run_on_cluster_settings(self.state.discovery_dataset, 'validation')

    def _init_predictions_df(self, dataset: Dataset):

        if len(self.state.label_config.get_labels_by_name()) > 0:
            predictions_df = dataset.get_metadata(self.state.label_config.get_labels_by_name(), return_df=True)
        else:
            predictions_df = pd.DataFrame(index=range(dataset.get_example_count()))

        predictions_df['example_id'] = dataset.get_example_ids()

        return predictions_df

    def _run_on_cluster_settings(self, dataset: Dataset, analysis_desc: str):

        predictions_df = self._init_predictions_df(dataset)
        clustering_items = {}

        for cl_setting in self.state.clustering_settings:
            cl_item, predictions_df = self._run_clustering_setting(dataset, cl_setting, predictions_df, analysis_desc)
            clustering_items[cl_setting.get_key()] = cl_item

        predictions_df.to_csv(self.state.predictions_paths[analysis_desc], index=False)
        self.state.clustering_items[analysis_desc] = clustering_items

    def _split_dataset(self):
        discovery_datasets, validation_datasets = HPUtil.split_data(self.state.dataset, self.state.split_config,
                                                                    self.state.result_path, self.state.label_config)

        self.state.discovery_dataset = discovery_datasets[0]
        self.state.validation_dataset = validation_datasets[0]

    def _setup_paths(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)
        self.state.predictions_paths = {'discovery': self.state.result_path / 'predictions_discovery.csv',
                                        'validation': self.state.result_path / 'predictions_validation.csv'}

    def _run_clustering_setting(self, dataset: Dataset, cl_setting: ClusteringSetting, predictions_df: pd.DataFrame,
                                analysis_desc: str) -> Tuple[ClusteringItem, pd.DataFrame]:

        cl_setting.path = PathBuilder.build(self.state.result_path / f"{analysis_desc}_{cl_setting.get_key()}")

        enc_dataset = DataEncoder.run(DataEncoderParams(dataset=dataset, encoder=cl_setting.encoder,
                                                        encoder_params=EncoderParams(model=cl_setting.encoder_params,
                                                                                     result_path=cl_setting.path,
                                                                                     pool_size=self.number_of_processes,
                                                                                     label_config=self.state.label_config,
                                                                                     learn_model=True,
                                                                                     encode_labels=False,
                                                                                     sequence_type=self.state.sequence_type,
                                                                                     region_type=self.state.region_type)))
        if cl_setting.dim_reduction_method is not None:
            enc_dataset.encoded_data.dimensionality_reduced_data = cl_setting.dim_reduction_method.fit_transform(enc_dataset)

        cl_setting.clustering_method.fit(enc_dataset)
        predictions = cl_setting.clustering_method.predict(enc_dataset)
        predictions_df[f'predictions_{cl_setting.get_key()}'] = predictions

        features = enc_dataset.encoded_data.examples if cl_setting.dim_reduction_method is None \
            else enc_dataset.encoded_data.dimensionality_reduced_data
        performance_paths = self._evaluate_clustering(predictions_df, cl_setting, features)

        cl_item = ClusteringItem(cl_setting=cl_setting, dataset=enc_dataset, predictions=predictions,
                                 external_performance=DataFrameWrapper(path=performance_paths['external']),
                                 internal_performance=DataFrameWrapper(path=performance_paths['internal']))

        self._run_reports(cl_item, analysis_desc)

        cl_item.dataset.encoded_data = None

        return cl_item, predictions_df

    def _evaluate_clustering(self, predictions: pd.DataFrame, cl_setting: ClusteringSetting, features) \
            -> Dict[str, Path]:

        result = {'internal': None, 'external': None}

        internal_performances = {}
        dense_features = features.toarray() if issparse(features) else features

        for metric in [m for m in self.state.metrics if is_internal(m)]:
            metric_fn = getattr(sklearn.metrics, metric)
            internal_performances[metric] = [metric_fn(dense_features,
                                                       predictions[f'predictions_{cl_setting.get_key()}'].values)]

        pd.DataFrame(internal_performances).to_csv(str(cl_setting.path / 'internal_performances.csv'), index=False)
        result['internal'] = cl_setting.path / 'internal_performances.csv'

        if self.state.label_config is not None and self.state.label_config.get_label_count() > 0:

            external_performances = {label: {} for label in self.state.label_config.get_labels_by_name()}

            for metric in [m for m in self.state.metrics if is_external(m)]:
                metric_fn = getattr(sklearn.metrics, metric)
                for label in self.state.label_config.get_labels_by_name():
                    external_performances[label][metric] = metric_fn(predictions[label].values,
                                                                     predictions[
                                                                         f'predictions_{cl_setting.get_key()}'].values)

            (pd.DataFrame(external_performances).reset_index().rename(columns={'index': 'metric'})
             .to_csv(str(cl_setting.path / 'external_performances.csv'), index=False))

            result['external'] = cl_setting.path / 'external_performances.csv'

        return result

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

    def _run_reports(self, cl_item: ClusteringItem, analysis_desc: str):
        report_path = PathBuilder.build(self.state.result_path / f'reports/{cl_item.cl_setting.get_key()}')
        for report in self.reports:
            if isinstance(report, EncodingReport):
                tmp_report = copy.deepcopy(report)
                tmp_report.result_path = report_path
                tmp_report.dataset = cl_item.dataset
                rep_result = tmp_report.generate_report()
                self.state.cl_item_report_results[analysis_desc][cl_item.cl_setting.get_key()]['encoding'].append(rep_result)

        if len(self.reports) > 0:
            gen_rep_count = sum(len(reports) for rep_type, reports in
                                self.state.cl_item_report_results[analysis_desc][cl_item.cl_setting.get_key()].items())
            print_log(f"{self.state.name}: generated {gen_rep_count} reports for setting "
                      f"{cl_item.cl_setting.get_key()} for {analysis_desc}.", True)

    def _init_report_result_structure(self):
        self.state.cl_item_report_results = {analysis_desc: {
            cl_setting.get_key(): {'encoding': []} for cl_setting in self.state.clustering_settings
        } for analysis_desc in ['discovery', 'validation']}
