import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
import plotly.express as px

from immuneML.IO.ml_method.ClusteringExporter import ClusteringExporter
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.clustering.StabilityLange import StabilityLange
from immuneML.hyperparameter_optimization.config.SampleConfig import SampleConfig
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.core.HPUtil import HPUtil
from immuneML.ml_metrics.ClusteringMetric import is_internal, is_external, get_search_criterion
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.Report import Report
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.Logger import print_log, log_memory_usage
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering import clustering_runner
from immuneML.workflows.instructions.clustering.ClusteringReportHandler import ClusteringReportHandler
from immuneML.workflows.instructions.clustering.ClusteringState import (
    ClusteringConfig, ClusteringState, ClusteringResultPerRun, StabilityConfig
)
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting
from immuneML.workflows.steps.DataSampler import DataSamplerParams, DataSampler


class ClusteringInstruction(Instruction):
    """

    Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
    clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. Finally, it
    provides options to include a set of reports to visualize the results.

    See also: :ref:`How to perform clustering analysis`.

    **Specification arguments:**

    - dataset (str): name of the dataset to be clustered

    - metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
      metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
      clusters can be compared against a list of predefined labels); some of the supported metrics include adjusted_rand_score,
      completeness_score, homogeneity_score, silhouette_score; for the full list, see scikit-learn's documentation of
      clustering metrics at https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster.

    - labels (list): an optional list of labels to use for external evaluation of clustering

    - sample_config (SampleConfig): configuration describing how to construct the data subsets to estimate different
      clustering settings' performance with different internal and external validation indices; with parameters
      `percentage`, `split_count`, `random_seed`:

    .. indent with spaces
    .. code-block:: yaml

        sample_config: # make 5 subsets with 80% of the data each
            split_count: 5
            percentage: 0.8
            random_seed: 42

    - stability_config (StabilityConfig): configuration describing how to compute clustering stability;
      currently, clustering stability is computed following approach by Lange et al. (2004) and only takes the number
      of repetitions as a parameter. Other strategies to compute clustering stability will be added in the future.

    .. indent with spaces
    .. code-block:: yaml

        stability_config:
            split_count: 5 # number of times to repeat clustering for stability estimation
            random_seed: 12

    - clustering_settings (list): a list where each element represents a :py:obj:`~immuneML.workflows.clustering.clustering_run_model.ClusteringSetting`; a combinations of encoding,
      optional dimensionality reduction algorithm, and the clustering algorithm that will be evaluated

    - random_labeling_count (int): number of random labelings to use for computing normalization value for stability
      assessment

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
                random_labeling_count: 5
                sample_config:
                    split_count: 5
                    percentage: 0.8
                    random_seed: 42
                stability_config:
                    split_count: 5
                    random_seed: 12
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
                 number_of_processes: int = None, sample_config: SampleConfig = None,
                 stability_config: StabilityConfig = None, random_labeling_count: int = None,
                 sequence_type: SequenceType = None, region_type: RegionType = None):

        config = ClusteringConfig(name=name, dataset=dataset, metrics=metrics, clustering_settings=clustering_settings,
                                  label_config=label_config, sample_config=sample_config, sequence_type=sequence_type,
                                  region_type=region_type, stability_config=stability_config,
                                  random_labeling_count=random_labeling_count)
        self.number_of_processes = number_of_processes
        self.state = ClusteringState(config=config, name=name)
        self.report_handler = ClusteringReportHandler(reports)

    def run(self, result_path: Path):
        """Main entry point: computes validation indices and estimates stability."""
        self.state.result_path = PathBuilder.build(result_path / self.state.config.name)

        self._fix_max_processes()

        # Step 1: Compute validation indices
        self._compute_validation_indices()

        # Step 2: Estimate clustering stability
        self._compute_stability()

        # Step 3: Refit the best settings on full dataset
        self._refit_best_settings_on_full_dataset()

        return self.state

    def _compute_validation_indices(self):
        """Compute internal and external validation indices across all datasets and settings."""
        self._setup_paths()

        # 1. Construct datasets (subsampling)
        datasets = self._construct_datasets()

        # 2. Run clustering settings on each dataset and collect predictions
        all_results = self._run_clustering_on_all_datasets(datasets)

        # 3. Aggregate and export index results
        self._aggregate_internal_indices(all_results)
        self._aggregate_external_indices(all_results)

        # 4. Run any additional clustering reports
        self.report_handler.run_clustering_reports(self.state)

    def _refit_best_settings_on_full_dataset(self):
        path = PathBuilder.build(self.state.result_path / "refitted_best_settings")

        best_settings = defaultdict(list)
        for metric_name, metric_path in self.state.metrics_performance_paths.items():
            best_setting = pd.read_csv(metric_path).drop(columns=['split_id']).mean(axis=0)

            if metric_name == 'stability_lange' or get_search_criterion(metric_name.split("__")[0]) == max:
                best_setting_key = best_setting.idxmax()
            else:
                best_setting_key = best_setting.idxmin()

            best_settings[best_setting_key].append(metric_path.stem)
            print_log(f"Best setting for metric {metric_path.stem} is {best_setting_key}.")

        predictions_df = self._init_predictions_df(self.state.config.dataset)
        for best_setting_key, per_metrics in best_settings.items():
            setting_path = path / best_setting_key
            cl_setting = self.state.config.get_cl_setting_by_key(best_setting_key)
            cl_item_res, predictions_df = clustering_runner.run_setting(dataset=self.state.config.dataset,
                                                    cl_setting=cl_setting,
                                                    path=setting_path, predictions_df=predictions_df,
                                                    metrics=[], label_config=self.state.config.label_config,
                                                    number_of_processes=self.number_of_processes,
                                                    sequence_type=self.state.config.sequence_type, evaluate=False,
                                                    region_type=self.state.config.region_type, state=self.state)
            cl_item_res.item.classifier = clustering_runner.train_cluster_classifier(cl_item_res.item)
            self.state.optimal_settings_on_discovery[best_setting_key] = cl_item_res

            # Export the best setting as a zip file
            zip_path = ClusteringExporter.export_zip(cl_item_res.item, setting_path, best_setting_key)
            self.state.best_settings_zip_paths[best_setting_key] = {'path': zip_path, 'metrics': per_metrics}
            logging.info(f"ClusteringInstruction: exported best setting {best_setting_key} to: {zip_path}")

        predictions_df.to_csv(path / "best_settings_predictions_full_dataset.csv", index=False)
        self.state.final_predictions_path = path / "best_settings_predictions_full_dataset.csv"


    def _construct_datasets(self) -> List[Dataset]:
        """Construct subsampled datasets for validation."""
        paths = [PathBuilder.build(f"{self.state.result_path}/split_{run_id + 1}/")
                 for run_id in range(self.state.config.sample_config.split_count)]

        self.state.subsampled_datasets = DataSampler.run(
            DataSamplerParams(self.state.config.dataset, self.state.config.sample_config, paths)
        )
        return self.state.subsampled_datasets

    def _run_clustering_on_all_datasets(self, datasets: List[Dataset]) -> List[Dict]:
        """Run every clustering setting on each dataset and collect results."""
        all_results = []

        for run_id, dataset in enumerate(datasets):
            print_log(f"Running clustering for split {run_id + 1}.")

            path = self.state.result_path / f"validation_indices/split_{run_id + 1}"
            predictions_df = self._init_predictions_df(dataset)

            clustering_items, predictions_df = clustering_runner.run_all_settings(
                dataset=dataset, path=path, run_id=run_id, predictions_df=predictions_df, state=self.state,
                report_handler=self.report_handler, number_of_processes=self.number_of_processes,
                sequence_type=self.state.config.sequence_type, region_type=self.state.config.region_type,
                clustering_settings=self.state.config.clustering_settings, metrics=self.state.config.metrics,
                label_config=self.state.config.label_config
            )

            predictions_df.to_csv(self.state.predictions_paths[run_id], index=False)

            cl_result = ClusteringResultPerRun(run_id, clustering_items)
            self.state.add_cl_result_per_run(run_id, cl_result)

            all_results.append({
                'run_id': run_id,
                'clustering_items': clustering_items,
                'predictions_df': predictions_df
            })

            log_memory_usage(f"discovery in split {run_id + 1}", f"Clustering instruction {self.state.name}")

        return all_results

    def _aggregate_internal_indices(self, all_results: List[Dict]):
        """
        Aggregate internal indices across all datasets and clustering settings.
        Produces one CSV per internal index, then generates boxplots.
        Creates ReportResult objects and adds them to state.
        """
        internal_metrics = [m for m in self.state.config.metrics if is_internal(m)]
        if not internal_metrics:
            return

        indices_path = PathBuilder.build(self.state.result_path / 'validation_indices' / 'internal')
        figures, tables = [], []

        for metric in internal_metrics:
            metric_data = self._collect_internal_metric_data(all_results, metric)

            csv_path = indices_path / f'{metric}.csv'
            metric_data.to_csv(csv_path, index=False)
            self.state.metrics_performance_paths[metric] = csv_path
            tables.append(ReportOutput(path=csv_path, name=f'{metric} values per split'))

            figures.append(self._create_internal_index_boxplot(metric_data, metric, indices_path))

        # Create ReportResult and add to state
        report_result = ReportResult(
            name=f'Internal Validation Indices',
            info=f'Internal validation indices ({", ".join(internal_metrics).replace(" ", "")}) computed '
                 f'across all clustering settings and data splits.',
            output_figures=figures,
            output_tables=tables
        )
        self.state.clustering_report_results.append(report_result)

    def _collect_internal_metric_data(self, all_results: List[Dict], metric: str) -> pd.DataFrame:
        """Collect internal metric values for all datasets and clustering settings."""
        rows = []

        for result in all_results:
            row = {'split_id': result['run_id'] + 1}

            for setting_key, cl_item_result in result['clustering_items'].items():
                internal_perf = cl_item_result.item.internal_performance
                if internal_perf is not None:
                    perf_df = internal_perf.get_df()
                    row[setting_key] = perf_df.loc[0, metric] if metric in perf_df.columns else None
                else:
                    row[setting_key] = None

            rows.append(row)

        return pd.DataFrame(rows)

    def _create_internal_index_boxplot(self, metric_data: pd.DataFrame, metric: str, output_path: Path) -> ReportOutput:
        """Create boxplot for internal index across clustering settings."""
        melted = metric_data.melt(id_vars=['split_id'], var_name='clustering_setting', value_name=metric)

        fig = px.box(melted, x='clustering_setting', y=metric,
                     labels={'clustering_setting': 'clustering setting', metric: metric},
                     color='clustering_setting', points='all',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(template='plotly_white', showlegend=False)

        plot_path = PlotlyUtil.write_image_to_file(fig, output_path / f'{metric}_boxplot.html', melted.shape[0])
        return ReportOutput(path=plot_path, name=f'{metric} across clustering settings')

    def _aggregate_external_indices(self, all_results: List[Dict]):
        """
        Aggregate external indices across all datasets, labels, and clustering settings.
        Produces one CSV per (label, external_index) combination, then generates boxplots.
        Creates ReportResult objects and adds them to state.
        """
        external_metrics = [m for m in self.state.config.metrics if is_external(m)]
        labels = self.state.config.label_config.get_labels_by_name() if self.state.config.label_config else []

        if not external_metrics or not labels:
            return

        indices_path = PathBuilder.build(self.state.result_path / 'validation_indices/external')
        figures, tables = [], []

        for label in labels:
            for metric in external_metrics:
                metric_data = self._collect_external_metric_data(all_results, label, metric)

                csv_path = indices_path / f'{metric}__{label}.csv'
                metric_data.to_csv(csv_path, index=False)
                self.state.metrics_performance_paths[f"{metric}__{label}"] = csv_path
                tables.append(ReportOutput(path=csv_path, name=f'{metric} values for label {label}'))

                figures.append(self._create_external_index_boxplot(metric_data, label, metric, indices_path))


        report_result = ReportResult(
            name=f'External Validation Indices',
            info=f'External validation indices ({", ".join(external_metrics).replace(" ", "")}) computed '
                 f'with respect to labels "{", ".join(labels).replace(" ", "")}" across all clustering '
                 f'settings and data splits.',
            output_figures=figures,
            output_tables=tables
        )
        self.state.clustering_report_results.append(report_result)

    def _collect_external_metric_data(self, all_results: List[Dict], label: str, metric: str) -> pd.DataFrame:
        """Collect external metric values for a specific label across all datasets and settings."""
        rows = []

        for result in all_results:
            row = {'split_id': result['run_id'] + 1}

            for setting_key, cl_item_result in result['clustering_items'].items():
                external_perf = cl_item_result.item.external_performance
                if external_perf is not None:
                    perf_df = external_perf.get_df()
                    # External performance is stored with metrics as rows, labels as columns
                    metric_row = perf_df[perf_df['metric'] == metric]
                    if not metric_row.empty and label in metric_row.columns:
                        row[setting_key] = metric_row[label].values[0]
                    else:
                        row[setting_key] = None
                else:
                    row[setting_key] = None

            rows.append(row)

        return pd.DataFrame(rows)

    def _create_external_index_boxplot(self, metric_data: pd.DataFrame, label: str, metric: str, output_path: Path) -> ReportOutput:
        """Create boxplot for external index per label, grouped by clustering setting."""
        melted = metric_data.melt(id_vars=['split_id'], var_name='clustering_setting', value_name=metric)

        fig = px.box(melted, x='clustering_setting', y=metric,
                     labels={'clustering_setting': 'clustering setting', metric: metric},
                     color='clustering_setting', points='all',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(template='plotly_white', showlegend=False)

        plot_path = PlotlyUtil.write_image_to_file(fig, output_path / f'{label}_{metric}_boxplot.html', melted.shape[0])
        return ReportOutput(path=plot_path, name=f'{metric} for label "{label}" across clustering settings')

    def _compute_stability(self):
        """Compute clustering stability using the Lange et al. approach."""
        discovery_datasets, tuning_datasets = HPUtil.split_data(
            self.state.config.dataset,
            SplitConfig(SplitType.RANDOM, self.state.config.stability_config.split_count, 0.5),
            PathBuilder.build(self.state.result_path / 'stability'),
            LabelConfiguration()
        )

        stab_lange = StabilityLange(
            discovery_datasets, tuning_datasets, self.state.config.clustering_settings,
            self.state.result_path / 'stability', self.number_of_processes, self.state.config.sequence_type,
            self.state.config.region_type, self.state.config.random_labeling_count
        )
        report_result, stability_path = stab_lange.run()
        self.state.clustering_report_results.append(report_result)
        self.state.metrics_performance_paths['stability_lange'] = stability_path

    def _setup_paths(self):
        """Initialize result paths."""
        self.state.predictions_paths = [
            PathBuilder.build(self.state.result_path / f"validation_indices/split_{run_id + 1}") / 'predictions.csv'
            for run_id in range(self.state.config.sample_config.split_count)
        ]

    def _init_predictions_df(self, dataset: Dataset) -> pd.DataFrame:
        """Initialize predictions DataFrame with labels and example IDs."""
        if len(self.state.config.label_config.get_labels_by_name()) > 0:
            predictions_df = dataset.get_metadata(self.state.config.label_config.get_labels_by_name(), return_df=True)
        else:
            predictions_df = pd.DataFrame(index=range(dataset.get_example_count()))

        predictions_df['example_id'] = dataset.get_example_ids()
        return predictions_df

    def _fix_max_processes(self):
        """Configure thread limits for parallelization."""
        if self.number_of_processes:
            try:
                import torch
                torch.set_num_threads(self.number_of_processes)
            except ImportError:
                pass

            os.environ["OMP_NUM_THREADS"] = str(self.number_of_processes)
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.number_of_processes)
            os.environ["MKL_NUM_THREADS"] = str(self.number_of_processes)