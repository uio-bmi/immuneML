import logging
from pathlib import Path
from typing import Callable, Tuple

import pandas as pd
from sklearn import metrics
from sklearn.metrics import jaccard_score

from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringRunner import get_features
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState
from immuneML.workflows.instructions.clustering.ValidationHandler import get_complementary_classifier


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
        self.desc = "Clustering Stability Report"

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
        self.result_path = PathBuilder.build(self.result_path / self.name)
        stability_results = []

        # For each data split
        for split_id in range(self.state.config.split_config.split_count):
            # For each clustering setting
            for setting in self.state.config.clustering_settings:
                setting_key = setting.get_key()

                # Get discovery and validation results
                discovery_result = self._get_cluster_result('discovery', split_id, setting_key)
                validation_result = self._get_cluster_result('method_based_validation', split_id, setting_key)

                if discovery_result and validation_result:
                    discovery_clusters = discovery_result.item.predictions
                    validation_clusters = validation_result.item.predictions

                    # Get encoded data
                    discovery_data = get_features(discovery_result.item.dataset, setting)
                    validation_data = get_features(validation_result.item.dataset, setting)

                    if discovery_data is not None and validation_data is not None:
                        try:
                            # Train classifier on discovery data
                            clf = get_complementary_classifier(setting)
                            clf.fit(discovery_data, discovery_clusters)
                            predicted_clusters = clf.predict(validation_data)

                            # Calculate stability score
                            stability_score = self.metric_fn(validation_clusters, predicted_clusters)

                            # Add additional information
                            result_entry = {
                                'split': split_id + 1,
                                'clustering_setting': setting_key,
                                self.metric: stability_score,
                                'n_clusters_discovery': len(set(discovery_clusters)),
                                'n_clusters_validation': len(set(validation_clusters))
                            }

                            stability_results.append(result_entry)
                        except Exception as e:
                            logging.warning(f"Error calculating stability for split {split_id + 1}, setting {setting_key}: {str(e)}")

        if not stability_results:
            logging.warning("No stability results could be calculated. Check if validation data is available.")
            return ReportResult(
                name=f"{self.desc} ({self.name})",
                info="No stability results could be calculated. Check if validation data is available."
            )

        # Create report outputs
        df = pd.DataFrame(stability_results)

        # Save detailed results table
        table_path = self.result_path / f"{self.result_name}.csv"
        df.to_csv(table_path, index=False)
        table_output = ReportOutput(path=table_path, name="Stability scores per setting")

        return ReportResult(
            name=f"{self.desc} ({self.name})",
            info=f"Analysis of clustering stability between discovery and validation datasets using {self.metric}.",
            output_tables=[table_output]
        )

    def _get_cluster_result(self, analysis_type: str, split_id: int, setting_key: str):
        """Get clustering result for a specific analysis type, split, and setting"""
        try:
            cl_result = self.state.clustering_items[split_id]
            if hasattr(cl_result, analysis_type):
                analysis_result = getattr(cl_result, analysis_type)
                if analysis_result and setting_key in analysis_result.items:
                    return analysis_result.items[setting_key]
        except (IndexError, AttributeError, KeyError) as e:
            logging.warning(f"Could not retrieve clustering result for {analysis_type}, split {split_id}, setting {setting_key}: {str(e)}")
        return None

    def check_prerequisites(self):
        run_report = True

        if self.state is None or not isinstance(self.state, ClusteringState):
            logging.warning(f"{self.__class__.__name__} requires a valid state object. Report will not be created.")
            run_report = False

        return run_report
