import logging
import numbers
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.neighbors import NearestCentroid

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.ml_methods.helper_methods.FurthestNeighborClassifier import FurthestNeighborClassifier
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringReportHandler import ClusteringReportHandler
from immuneML.workflows.instructions.clustering.ClusteringRunner import ClusteringRunner, get_features, encode_dataset
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringConfig, ClusteringState, \
    ClusteringResultPerRun, ClusteringItemResult
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting, ClusteringItem, \
    DataFrameWrapper


class ValidationHandler:
    """Handles different validation strategies for clustering."""

    def __init__(self, config: ClusteringConfig, runner: ClusteringRunner, report_handler: ClusteringReportHandler,
                 num_of_processes: int):
        self.config = config
        self.runner = runner
        self.report_handler = report_handler
        self.number_of_processes = num_of_processes

    def run_method_based_validation(self, dataset: Dataset, run_id: int, path: Path, predictions_df: pd.DataFrame,
                                    state: ClusteringState):
        """Run method-based validation."""
        print_log(f"Running method-based validation for run {run_id + 1}")
        cl_items, predictions_df = self.runner.run_all_settings(dataset, 'method_based_validation', path,
                                                                run_id, predictions_df, state)
        state.add_cl_result_per_run(run_id, 'method_based_validation',
                                    ClusteringResultPerRun(run_id, 'method_based_validation', cl_items))
        predictions_df.to_csv(state.predictions_paths[run_id]['method_based_validation'], index=False)
        print_log(f"Method-based validation for run {run_id + 1} completed.")
        return state

    def run_result_based_validation(self, dataset: Dataset, run_id: int, path: Path, predictions_df: pd.DataFrame,
                                    state: ClusteringState):
        """Run result-based validation by training a classifier on discovery clusters."""
        cl_items = {}
        analysis_desc = 'result_based_validation'
        print_log(f"Running result-based validation for run {run_id + 1}")

        for cl_setting in self.config.clustering_settings:
            # Get discovery data clustering results
            discovery_item = state.clustering_items[run_id].discovery.items[cl_setting.get_key()].item
            discovery_dataset = discovery_item.dataset

            # Train classifier on discovery data using clusters as labels
            classifier = self._train_cluster_classifier(discovery_item, cl_setting, discovery_dataset)
            cl_setting.path = PathBuilder.build(path / f"{cl_setting.get_key()}")

            # Apply classifier to validation data
            cl_item, predictions_df = self._apply_cluster_classifier(
                dataset=dataset,
                cl_setting=cl_setting,
                classifier=classifier,
                predictions_df=predictions_df,
                analysis_desc=analysis_desc,
                run_id=run_id,
                path=cl_setting.path,
                encoder=discovery_item.encoder
            )

            report_results = self.report_handler.run_item_reports(cl_item, analysis_desc, run_id, cl_setting.path,
                                                                  state)
            cl_items[cl_setting.get_key()] = ClusteringItemResult(cl_item, report_results)

        predictions_df.to_csv(state.predictions_paths[run_id][analysis_desc], index=False)
        state.add_cl_result_per_run(run_id, analysis_desc, ClusteringResultPerRun(run_id, analysis_desc, cl_items))

        print_log(f"Result-based validation for run {run_id + 1} completed.")
        return state

    def _train_cluster_classifier(self, discovery_clusters: ClusteringItem, cl_setting: ClusteringSetting,
                                  discovery_dataset: Dataset):
        """Train a classifier using discovery data clusters as labels."""
        print_log(f"Training clustering-based classifier for {cl_setting.get_key()} for result-based validation.")
        classifier = get_complementary_classifier(cl_setting)

        # Get features and cluster assignments from discovery data
        features = get_features(discovery_dataset, cl_setting)

        if len(list(set(discovery_clusters.predictions))) == 1:
            return discovery_clusters.predictions[0]
        else:
            classifier.fit(features, discovery_clusters.predictions)
            return classifier

    def _apply_cluster_classifier(self, dataset: Dataset, cl_setting: ClusteringSetting, classifier,
                                  predictions_df: pd.DataFrame, analysis_desc: str, run_id: int, path: Path,
                                  encoder: DatasetEncoder) -> Tuple[ClusteringItem, pd.DataFrame]:
        """Apply trained classifier to validation data."""

        enc_dataset = encode_dataset(dataset, cl_setting, self.number_of_processes, self.config.label_config,
                                     learn_model=False, sequence_type=self.config.sequence_type,
                                     region_type=self.config.region_type, encoder=encoder)
        features = get_features(enc_dataset, cl_setting)

        if isinstance(classifier, numbers.Number):
            predictions = [classifier] * dataset.get_example_count()
            logging.warning(f"Only one cluster found in discovery data. Assigning all validation data to "
                            f"cluster {classifier}.")
        else:
            predictions = classifier.predict(features)

        predictions_df[f'predictions_{cl_setting.get_key()}'] = predictions

        # Evaluate clustering
        performance_paths = self.runner.evaluate_clustering(predictions_df, cl_setting, features)

        cl_item = ClusteringItem(
            cl_setting=cl_setting,
            dataset=enc_dataset,
            predictions=predictions,
            external_performance=DataFrameWrapper(path=performance_paths['external']),
            internal_performance=DataFrameWrapper(path=performance_paths['internal'])
        )

        print_log(f"Result-based validation for clustering setting {cl_setting.get_key()} finished.")

        return cl_item, predictions_df


def get_complementary_classifier(cl_setting: ClusteringSetting):
    """
    Returns a complementary classifier based on the clustering method.

    Args:
        cl_setting: ClusteringSetting object containing the clustering method configuration

    Returns:
        An instance of the appropriate classifier; NearestCentroid if no matches are found
    """
    clustering_method = cl_setting.clustering_method
    method_name = clustering_method.__class__.__name__

    if method_name == 'KMeans':
        return NearestCentroid()
    elif method_name == 'AgglomerativeClustering':
        if hasattr(clustering_method.model, 'linkage'):
            if clustering_method.model.linkage == 'ward':
                return NearestCentroid()
            elif clustering_method.model.linkage == 'complete':
                if cl_setting.encoder.__class__.__name__ == 'TCRdistEncoder':
                    return FurthestNeighborClassifier(metric='precomputed')
                else:
                    return FurthestNeighborClassifier(metric='euclidean')

    return NearestCentroid()
