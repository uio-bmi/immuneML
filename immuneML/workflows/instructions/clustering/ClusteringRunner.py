import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import issparse

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.ml_metrics.ClusteringMetric import is_external, is_internal
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringReportHandler import ClusteringReportHandler
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringConfig, ClusteringState
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem, DataFrameWrapper, \
    ClusteringSetting
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


class ClusteringRunner:
    """Handles core clustering operations like fitting, prediction and evaluation."""

    def __init__(self, config: ClusteringConfig, n_processes: int, report_handler: ClusteringReportHandler):
        self.config = config
        self.report_handler = report_handler
        self.number_of_processes = n_processes

    def run_all_settings(self, dataset: Dataset, analysis_desc: str, path: Path, run_id: int,
                         predictions_df: pd.DataFrame, state: ClusteringState):
        clustering_items = {}

        for cl_setting in self.config.clustering_settings:
            cl_item, predictions_df = self.run_setting(dataset, cl_setting, analysis_desc, path, run_id,
                                                       predictions_df, state)
            clustering_items[cl_setting.get_key()] = cl_item

        return clustering_items, predictions_df

    def run_setting(self, dataset: Dataset, cl_setting: ClusteringSetting, analysis_desc: str, path: Path,
                    run_id: int, predictions_df: pd.DataFrame, state: ClusteringState) -> Tuple[ClusteringItem, pd.DataFrame]:
        cl_setting.path = PathBuilder.build(path / f"{cl_setting.get_key()}")

        # Encode data
        encoder = cl_setting.encoder if analysis_desc == 'discovery' \
            else state.clustering_items[run_id]['discovery'][cl_setting.get_key()].encoder
        enc_dataset = self.encode_dataset(dataset, cl_setting, learn_model=analysis_desc == 'discovery',
                                          encoder=encoder)

        # Run clustering
        predictions = self._fit_and_predict(enc_dataset, cl_setting)
        predictions_df[f'predictions_{cl_setting.get_key()}'] = predictions

        # Evaluate results
        features = self.get_features(enc_dataset, cl_setting)
        performance_paths = self.evaluate_clustering(predictions_df, cl_setting, features)

        cl_item = ClusteringItem(
            cl_setting=cl_setting,
            dataset=enc_dataset,
            predictions=predictions,
            encoder=encoder,
            external_performance=DataFrameWrapper(path=performance_paths['external']),
            internal_performance=DataFrameWrapper(path=performance_paths['internal'])
        )

        self.report_handler.run_item_reports(cl_item, analysis_desc, run_id, cl_setting.path, state)

        return cl_item, predictions_df

    def encode_dataset(self, dataset: Dataset, cl_setting: ClusteringSetting, learn_model: bool = True,
                       encoder: DatasetEncoder = None) -> Dataset:
        """Handle dataset encoding and dimensionality reduction."""
        enc_params = EncoderParams(model=cl_setting.encoder_params, result_path=cl_setting.path,
                                   pool_size=self.number_of_processes, label_config=self.config.label_config,
                                   learn_model=learn_model, encode_labels=False,
                                   sequence_type=self.config.sequence_type,
                                   region_type=self.config.region_type)
        enc_dataset = DataEncoder.run(DataEncoderParams(dataset=dataset,
                                                        encoder=cl_setting.encoder if encoder is None else encoder,
                                                        encoder_params=enc_params))

        if cl_setting.dim_reduction_method:
            enc_dataset.encoded_data.dimensionality_reduced_data = cl_setting.dim_reduction_method.fit_transform(
                enc_dataset)

        return enc_dataset

    def get_features(self, dataset: Dataset, cl_setting: ClusteringSetting):
        """Get features from encoded dataset."""
        return dataset.encoded_data.examples if cl_setting.dim_reduction_method is None \
            else dataset.encoded_data.dimensionality_reduced_data

    def _fit_and_predict(self, dataset: Dataset, cl_setting: ClusteringSetting) -> np.ndarray:
        """Fit clustering method and get predictions."""
        cl_setting.clustering_method.fit(dataset)
        return cl_setting.clustering_method.predict(dataset)

    def evaluate_clustering(self, predictions: pd.DataFrame, cl_setting: ClusteringSetting, features) \
            -> Dict[str, Path]:

        result = {'internal': None, 'external': None}

        internal_performances = {}
        dense_features = features.toarray() if issparse(features) else features

        for metric in [m for m in self.config.metrics if is_internal(m)]:
            try:
                metric_fn = getattr(sklearn.metrics, metric)
                internal_performances[metric] = [metric_fn(dense_features,
                                                           predictions[f'predictions_{cl_setting.get_key()}'].values)]
            except ValueError as e:
                logging.info(f"Error calculating metric {metric}: {e}")
                internal_performances[metric] = [np.nan]

        pd.DataFrame(internal_performances).to_csv(str(cl_setting.path / 'internal_performances.csv'), index=False)
        result['internal'] = cl_setting.path / 'internal_performances.csv'

        if self.config.label_config is not None and self.config.label_config.get_label_count() > 0:

            external_performances = {label: {} for label in self.config.label_config.get_labels_by_name()}

            for metric in [m for m in self.config.metrics if is_external(m)]:

                metric_fn = getattr(sklearn.metrics, metric)
                for label in self.config.label_config.get_labels_by_name():
                    try:
                        external_performances[label][metric] = metric_fn(predictions[label].values,
                                                                         predictions[
                                                                             f'predictions_{cl_setting.get_key()}'].values)
                    except ValueError as e:
                        logging.info(f"Error calculating metric {metric}: {e}")
                        external_performances[label][metric] = np.nan

                metric_fn = getattr(sklearn.metrics, metric)
                for label in self.config.label_config.get_labels_by_name():
                    external_performances[label][metric] = metric_fn(predictions[label].values,
                                                                     predictions[
                                                                         f'predictions_{cl_setting.get_key()}'].values)

            (pd.DataFrame(external_performances).reset_index().rename(columns={'index': 'metric'})
             .to_csv(str(cl_setting.path / 'external_performances.csv'), index=False))

            result['external'] = cl_setting.path / 'external_performances.csv'

        return result
