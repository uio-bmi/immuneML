from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringReportHandler import ClusteringReportHandler
from immuneML.workflows.instructions.clustering.ClusteringRunner import ClusteringRunner, get_features
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringConfig, ClusteringState
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting, ClusteringItem, \
    DataFrameWrapper


class ValidationHandler:
    """Handles different validation strategies for clustering."""

    def __init__(self, config: ClusteringConfig, runner: ClusteringRunner, report_handler: ClusteringReportHandler):
        self.config = config
        self.runner = runner
        self.report_handler = report_handler

    def run_method_based_validation(self, dataset: Dataset, run_id: int, path: Path, predictions_df: pd.DataFrame,
                                    state: ClusteringState):
        """Run method-based validation."""
        cl_items, predictions_df = self.runner.run_all_settings(dataset, 'method_based_validation', path,
                                                                run_id, predictions_df, state)
        state.clustering_items[run_id]['method_based_validation'] = cl_items
        predictions_df.to_csv(state.predictions_paths[run_id]['method_based_validation'], index=False)

        return state

    def run_result_based_validation(self, dataset: Dataset, run_id: int, path: Path, predictions_df: pd.DataFrame,
                                    state: ClusteringState):
        """Run result-based validation by training a classifier on discovery clusters."""
        clustering_items = {}

        for cl_setting in self.config.clustering_settings:
            # Get discovery data clustering results
            discovery_item = state.clustering_items[run_id]['discovery'][cl_setting.get_key()]

            # Train classifier on discovery data using clusters as labels
            classifier = self._train_cluster_classifier(discovery_item, cl_setting)
            cl_setting.path = PathBuilder.build(path / f"{cl_setting.get_key()}")

            # Apply classifier to validation data
            cl_item, predictions_df = self._apply_cluster_classifier(
                dataset=dataset,
                cl_setting=cl_setting,
                classifier=classifier,
                predictions_df=predictions_df,
                analysis_desc='result_based_validation',
                run_id=run_id,
                path=cl_setting.path,
                encoder=discovery_item.encoder
            )

            clustering_items[cl_setting.get_key()] = cl_item
            state = self.report_handler.run_item_reports(cl_item, "result_based_validation", run_id, cl_setting.path,
                                                         state)

        predictions_df.to_csv(state.predictions_paths[run_id]['result_based_validation'], index=False)
        state.clustering_items[run_id]['result_based_validation'] = clustering_items

        return state

    def _train_cluster_classifier(self, discovery_clusters: ClusteringItem, cl_setting: ClusteringSetting):
        """Train a classifier using discovery data clusters as labels."""
        classifier = get_complementary_classifier(cl_setting)

        # Get features and cluster assignments from discovery data
        features = (discovery_clusters.dataset.encoded_data.examples
                    if discovery_clusters.dataset.encoded_data.dimensionality_reduced_data is None
                    else discovery_clusters.dataset.encoded_data.dimensionality_reduced_data)

        classifier.fit(features, discovery_clusters.predictions)
        return classifier

    def _apply_cluster_classifier(self, dataset: Dataset,
                                  cl_setting: ClusteringSetting,
                                  classifier,
                                  predictions_df: pd.DataFrame,
                                  analysis_desc: str,
                                  run_id: int, path: Path, encoder: DatasetEncoder) -> Tuple[
        ClusteringItem, pd.DataFrame]:
        """Apply trained classifier to validation data."""
        enc_dataset = self.runner.encode_dataset(dataset, cl_setting, learn_model=False, encoder=encoder)
        features = get_features(enc_dataset, cl_setting)

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
                def custom_weights(distances):
                    return np.eye(len(distances))[:, -1]

                return KNeighborsClassifier(weights=custom_weights)

    return NearestCentroid()
