import logging
from pathlib import Path
from typing import List, Dict, Callable, Tuple

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score, jaccard_score

from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.workflows.instructions.clustering.ClusteringRunner import get_features
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState
from immuneML.workflows.instructions.clustering.ValidationHandler import ValidationHandler, get_complementary_classifier
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting


class ClusteringStabilityReport(ClusteringReport):
    """
    Report that analyzes clustering stability by comparing results between discovery and validation datasets.
    The comparison uses a classifier-based approach where:
    1. A classifier is trained on discovery data using cluster assignments as labels
    2. Cluster assignments are predicted for validation data
    3. Predictions are compared with actual validation clustering results using the specified similarity metric

    This report can be used with the Clustering instruction under 'reports'.

    **Specification arguments:**

    - metric (str): Name of any clustering evaluation metric from sklearn.metrics that compares two sets of labels
      (e.g., adjusted_rand_score, jaccard_score, adjusted_mutual_info_score, normalized_mutual_info_score).
      If an invalid metric name is provided, defaults to adjusted_rand_score.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_clustering_instruction:
            type: Clustering
            reports:
                my_stability_report:
                    ClusteringStabilityReport:
                        metric: jaccard_score

    """

    DEFAULT_METRIC = "adjusted_rand_score"

    @classmethod
    def build_object(cls, **kwargs):
        location = "ClusteringStabilityReport"
        name = kwargs["name"] if "name" in kwargs else None
        similarity_metric = kwargs.get("similarity_metric", "ari")

        ParameterValidator.assert_type_and_value(similarity_metric, str, location, "similarity_metric")
        ParameterValidator.assert_in_valid_list(similarity_metric.lower(), ["ari", "jaccard"], location,
                                                "similarity_metric")

        return ClusteringStabilityReport(name=name, similarity_metric=similarity_metric)

    def __init__(self, similarity_metric: str, name: str = None, state: ClusteringState = None,
                 result_path: Path = None, number_of_processes: int = 1):
        super().__init__(name=name, result_path=result_path, number_of_processes=number_of_processes)
        self.state = state
        self.metric_fn, self.metric = self._get_similarity_metric(similarity_metric)
        self.result_name = "clustering_stability_analysis"

    def _get_similarity_metric(self, metric_name: str) -> Tuple[Callable, str]:
        try:
            metric_func = getattr(metrics, metric_name)
            if callable(metric_func):
                if metric_name == 'jaccard_score':
                    def jaccard_weighted(x, y):
                        return jaccard_score(x, y, average='weighted')

                    metric_func = jaccard_weighted
                    logging.info(f"Using Jaccard metric for clustering stability analysis with weighted average")
                return metric_func, metric_name
        except AttributeError:
            logging.warning(f"Invalid similarity metric '{metric_name}', defaulting to {self.DEFAULT_METRIC}")
            return getattr(metrics, self.DEFAULT_METRIC), self.DEFAULT_METRIC

    def _generate(self) -> ReportResult:
        """Generate the stability analysis report"""
        stability_results = []

        # For each data split
        for split_id in range(len(self.state.discovery_datasets)):
            # For each clustering setting
            for setting in self.state.config.clustering_settings:
                setting_key = setting.get_key()

                # Get discovery and validation results
                discovery_clusters = self._get_cluster_assignments('discovery', split_id, setting_key)
                validation_clusters = self._get_cluster_assignments('method_based_validation', split_id, setting_key)

                if discovery_clusters is not None and validation_clusters is not None:
                    # Get encoded data
                    discovery_data = self._get_encoded_data('discovery', split_id, setting)
                    validation_data = self._get_encoded_data('method_based_validation', split_id, setting)

                    if discovery_data is not None and validation_data is not None:
                        clf = get_complementary_classifier(setting)
                        clf.fit(discovery_data, discovery_clusters)
                        predicted_clusters = clf.predict(validation_data)

                        # Calculate stability score
                        stability_score = self.metric_fn(validation_clusters, predicted_clusters)

                        stability_results.append({
                            'split': split_id + 1,
                            'clustering_setting': setting_key,
                            self.metric: stability_score
                        })

        # Create report outputs
        df = pd.DataFrame(stability_results)

        # Save results table
        table_path = self.result_path / f"{self.result_name}.csv"
        df.to_csv(table_path, index=False)
        table_output = ReportOutput(path=table_path, name="Stability scores per setting")

        return ReportResult(
            name=self.name,
            info="Analysis of clustering stability between discovery and validation datasets.",
            output_tables=[table_output],
        )

    def _get_cluster_assignments(self, analysis_type: str, split_id: int, setting_key: str):
        try:
            return self.state.clustering_items[split_id][analysis_type][setting_key].predictions
        except (KeyError, AttributeError):
            return None

    def _get_encoded_data(self, analysis_type: str, split_id: int, setting: ClusteringSetting):
        try:
            return get_features(self.state.clustering_items[split_id][analysis_type][setting.get_key()].dataset,
                                setting)
        except (KeyError, AttributeError):
            return None

    def _create_report_output(self, df: pd.DataFrame) -> str:
        """Creates a text summary of the stability analysis results"""
        output = []
        output.append(f"Clustering Stability Analysis using {self.metric.upper()} metric\n")

        # Overall statistics
        mean_stability = df['stability_score'].mean()
        std_stability = df['stability_score'].std()
        output.append(f"Overall mean stability score: {mean_stability:.3f} (±{std_stability:.3f})\n")

        # Per setting statistics
        for setting in df['clustering_setting'].unique():
            setting_data = df[df['clustering_setting'] == setting]
            output.append(f"\nClustering setting: {setting}")
            output.append(f"Mean stability score: {setting_data['stability_score'].mean():.3f}")
            output.append(f"Std stability score: {setting_data['stability_score'].std():.3f}")
            output.append(f"Average number of clusters (discovery): {setting_data['n_clusters_discovery'].mean():.1f}")
            output.append(
                f"Average number of clusters (validation): {setting_data['n_clusters_validation'].mean():.1f}")

        return "\n".join(output)

    def check_prerequisites(self):
        run_report = True

        if self.state is None or not isinstance(self.state, ClusteringState):
            logging.warning(f"{self.__class__.__name__} requires a valid state object. Report will not be created.")
            run_report = False

        return run_report
