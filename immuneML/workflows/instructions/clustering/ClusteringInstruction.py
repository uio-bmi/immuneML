import copy
import os
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.reports.Report import Report
from immuneML.util.Logger import print_log, log_memory_usage
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.ClusteringReportHandler import ClusteringReportHandler
from immuneML.workflows.instructions.clustering.ClusteringRunner import ClusteringRunner
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringConfig, ClusteringState, \
    ClusteringResultPerRun
from immuneML.workflows.instructions.clustering.ValidationHandler import ValidationHandler
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting


class ClusteringInstruction(Instruction):
    """

    Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
    clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. The dataset is split
    into discovery and validation datasets and the clustering results are reported on both. Finally, it
    provides options to include a set of reports to visualize the results.

    See also: :ref:`How to perform clustering analysis`

    For more details on choosing the clustering algorithm and its hyperparameters, see the paper:
    Ullmann, T., Hennig, C., & Boulesteix, A.-L. (2022). Validation of cluster analysis results on validation
    data: A systematic framework. WIREs Data Mining and Knowledge Discovery, 12(3), e1444.
    https://doi.org/10.1002/widm.1444


    **Specification arguments:**

    - dataset (str): name of the dataset to be clustered

    - metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
      metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
      clusters can be compared against a list of predefined labels); some of the supported metrics include adjusted_rand_score,
      completeness_score, homogeneity_score, silhouette_score; for the full list, see scikit-learn's documentation of
      clustering metrics at https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster.

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
            split_count: 3 # repeat the random split 3 times -> 3 discovery and 3 validation datasets

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

    - validation_type (list): a list of validation types to use for comparison of clustering algorithms and encodings;
      it can be method_based and/or result_based

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
                validation_type: [method_based, result_based]
                split_config:
                    split_count: 1
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
                 region_type: RegionType = None, validation_type: List[str] = None):

        config = ClusteringConfig(name=name, dataset=dataset, metrics=metrics, clustering_settings=clustering_settings,
                                  label_config=label_config, split_config=split_config, sequence_type=sequence_type,
                                  region_type=region_type, validation_type=validation_type)
        self.number_of_processes = number_of_processes
        self.state = ClusteringState(config=config, name=name)
        self.report_handler = ClusteringReportHandler(reports)
        self.cl_runner = ClusteringRunner(self.state.config, self.number_of_processes, self.report_handler)
        self.validation_handler = ValidationHandler(self.state.config, self.cl_runner, self.report_handler,
                                                    self.number_of_processes)

    def run(self, result_path: Path):
        """Execute the clustering instruction workflow."""
        self._fix_max_processes()
        self._setup_paths(result_path)
        self._split_dataset()

        for run_id in range(self.state.config.split_config.split_count):

            print_log(f"Running clustering for split {run_id + 1}.")

            path = self.state.result_path / f"split_{run_id + 1}"

            # Run discovery
            self._run_discovery(run_id, path / 'discovery')

            log_memory_usage(f"discovery in split {run_id + 1}", f"Clustering instruction {self.state.name}")

            # Run validations
            print_log(f"Running validation for split {run_id + 1}.")

            predictions_df = self._init_predictions_df(self.state.validation_datasets[run_id])

            if "method_based" in self.state.config.validation_type:
                self.state = self.validation_handler.run_method_based_validation(
                    self.state.validation_datasets[run_id], run_id, PathBuilder.build(path / 'method_based_validation'),
                    copy.deepcopy(predictions_df), self.state)

                log_memory_usage(f"method-based validation in split {run_id + 1}",
                                 f"Clustering instruction {self.state.name}")

            if "result_based" in self.state.config.validation_type:
                self.state = self.validation_handler.run_result_based_validation(
                    self.state.validation_datasets[run_id], run_id, PathBuilder.build(path / "result_based_validation"),
                    copy.deepcopy(predictions_df), self.state)

                log_memory_usage(f"result-based validation in split {run_id + 1}",
                                 f"Clustering instruction {self.state.name}")

            print_log(f"Clustering for split {run_id + 1} finished.")

        self.report_handler.run_clustering_reports(self.state)
        return self.state

    def _run_discovery(self, run_id: int, path: Path):
        """Run clustering on discovery data."""

        print_log("Running clustering on discovery data.")

        dataset = self.state.discovery_datasets[run_id]
        analysis_desc = 'discovery'

        predictions_df = self._init_predictions_df(dataset)
        clustering_items, predictions_df = self.cl_runner.run_all_settings(dataset, analysis_desc, path, run_id,
                                                                           predictions_df, self.state)

        predictions_df.to_csv(self.state.predictions_paths[run_id][analysis_desc], index=False)
        cl_result = ClusteringResultPerRun(run_id, analysis_desc, clustering_items)
        self.state.add_cl_result_per_run(run_id, analysis_desc, cl_result)

        print_log("Clustering on discovery data finished.")

    def _split_dataset(self):
        self.state.discovery_datasets, self.state.validation_datasets = HPUtil.split_data(
            dataset=self.state.config.dataset,
            split_config=self.state.config.split_config,
            path=self.state.result_path,
            label_config=self.state.config.label_config)

    def _setup_paths(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.config.name)
        self.state.predictions_paths = [
            {'discovery': PathBuilder.build(
                self.state.result_path / f"split_{run_id + 1}") / 'predictions_discovery.csv',
             'result_based_validation': self.state.result_path / f'split_{run_id + 1}/predictions_result_based_validation.csv',
             'method_based_validation': self.state.result_path / f'split_{run_id + 1}/predictions_method_based_validation.csv'}
            for run_id in range(self.state.config.split_config.split_count)]

    def _init_predictions_df(self, dataset: Dataset) -> pd.DataFrame:

        if len(self.state.config.label_config.get_labels_by_name()) > 0:
            predictions_df = dataset.get_metadata(self.state.config.label_config.get_labels_by_name(), return_df=True)
        else:
            predictions_df = pd.DataFrame(index=range(dataset.get_example_count()))

        predictions_df['example_id'] = dataset.get_example_ids()

        return predictions_df

    def _fix_max_processes(self):

        if self.number_of_processes:

            try:
                import torch
                torch.set_num_threads(self.number_of_processes)
            except ImportError as e:
                pass

            os.environ["OMP_NUM_THREADS"] = str(self.number_of_processes)
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.number_of_processes)
            os.environ["MKL_NUM_THREADS"] = str(self.number_of_processes)
