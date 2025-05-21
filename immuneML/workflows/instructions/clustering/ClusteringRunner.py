import copy
import logging
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import issparse

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_metrics.ClusteringMetric import is_external, is_internal
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringReportHandler import ClusteringReportHandler
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringConfig, ClusteringState, \
    ClusteringItemResult
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
                    run_id: int, predictions_df: pd.DataFrame, state: ClusteringState) \
            -> Tuple[ClusteringItemResult, pd.DataFrame]:

        print_log(f"Running clustering setting {cl_setting.get_key()}")

        cl_setting.path = PathBuilder.build(path / f"{cl_setting.get_key()}")

        learn_model = analysis_desc == 'discovery' or analysis_desc == 'method_based_validation'

        # Encode data
        encoder = copy.deepcopy(cl_setting.encoder) if learn_model \
            else state.clustering_items[run_id].discovery.get_cl_item(cl_setting).encoder
        enc_dataset = encode_dataset(dataset, cl_setting, self.number_of_processes, self.config.label_config,
                                     learn_model, self.config.sequence_type, self.config.region_type,
                                     encoder)

        # Run clustering
        predictions = self._fit_and_predict(enc_dataset, cl_setting)
        predictions_df[f'predictions_{cl_setting.get_key()}'] = predictions

        print_log(f"{cl_setting.get_key()}: clustering method fitted and predictions made.")

        # Evaluate results
        features = get_features(enc_dataset, cl_setting)
        performance_paths = self.evaluate_clustering(predictions_df, cl_setting, features)

        cl_item = ClusteringItem(
            cl_setting=cl_setting,
            dataset=enc_dataset,
            predictions=predictions,
            encoder=encoder,
            external_performance=DataFrameWrapper(path=performance_paths['external']),
            internal_performance=DataFrameWrapper(path=performance_paths['internal'])
        )

        report_results = self.report_handler.run_item_reports(cl_item, analysis_desc, run_id, cl_setting.path, state)

        print_log(f"Clustering setting {cl_setting.get_key()} finished.")

        return ClusteringItemResult(cl_item, report_results), predictions_df

    def _fit_and_predict(self, dataset: Dataset, cl_setting: ClusteringSetting) -> np.ndarray:
        """Fit clustering method and get predictions."""
        if hasattr(cl_setting.clustering_method, 'fit_predict'):
            return cl_setting.clustering_method.fit_predict(dataset)
        else:
            cl_setting.clustering_method.fit(dataset)
            return cl_setting.clustering_method.predict(dataset)

    def _eval_internal_metrics(self, predictions: pd.DataFrame, cl_setting: ClusteringSetting, features) -> Path:
        internal_performances = {}
        dense_features = features.toarray() if issparse(features) else features

        for metric in [m for m in self.config.metrics if is_internal(m)]:
            try:
                metric_fn = getattr(sklearn.metrics, metric)
                internal_performances[metric] = [metric_fn(dense_features,
                                                           predictions[f'predictions_{cl_setting.get_key()}'].values)]
            except ValueError as e:
                logging.warning(f"Error calculating metric {metric}: {e}")
                internal_performances[metric] = [np.nan]

        pd.DataFrame(internal_performances).to_csv(str(cl_setting.path / 'internal_performances.csv'), index=False)
        return cl_setting.path / 'internal_performances.csv'

    def _eval_external_metrics(self, predictions: pd.DataFrame, cl_setting: ClusteringSetting) -> Union[Path, None]:
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
                        logging.warning(f"Error calculating metric {metric}: {e}")
                        external_performances[label][metric] = np.nan

            (pd.DataFrame(external_performances).reset_index().rename(columns={'index': 'metric'})
             .to_csv(str(cl_setting.path / 'external_performances.csv'), index=False))

            return cl_setting.path / 'external_performances.csv'
        else:
            return None

    def evaluate_clustering(self, predictions: pd.DataFrame, cl_setting: ClusteringSetting, features) \
            -> Dict[str, Path]:

        return {'internal': self._eval_internal_metrics(predictions, cl_setting, features),
                'external': self._eval_external_metrics(predictions, cl_setting)}


def get_features(dataset: Dataset, cl_setting: ClusteringSetting):
    """Get features from encoded dataset."""
    return dataset.encoded_data.examples if cl_setting.dim_reduction_method is None \
        else dataset.encoded_data.dimensionality_reduced_data


def encode_dataset(dataset: Dataset, cl_setting: ClusteringSetting, number_of_processes: int,
                   label_config: LabelConfiguration, learn_model: bool, sequence_type: SequenceType,
                   region_type: RegionType, encoder: DatasetEncoder = None):
    def dict_to_sorted_tuple(d: dict) -> tuple:
        return tuple(sorted(d.items(), key=lambda tup: tup[0])) if d is not None else None

    return CacheHandler.memo_by_params(
        (dataset.identifier, dict_to_sorted_tuple(cl_setting.encoder_params), label_config.get_labels_by_name(),
         sequence_type.name, region_type.name, cl_setting.dim_red_name, learn_model,
         dict_to_sorted_tuple(cl_setting.dim_red_params)),
        lambda: encode_dataset_internal(dataset, cl_setting, number_of_processes, label_config,
                                        learn_model, sequence_type, region_type, encoder))


def encode_dataset_internal(dataset: Dataset, cl_setting: ClusteringSetting, number_of_processes: int,
                            label_config: LabelConfiguration, learn_model: bool, sequence_type: SequenceType,
                            region_type: RegionType, encoder: DatasetEncoder = None):
    enc_params = EncoderParams(model=cl_setting.encoder_params, result_path=cl_setting.path,
                               pool_size=number_of_processes, label_config=label_config,
                               learn_model=learn_model, encode_labels=False,
                               sequence_type=sequence_type,
                               region_type=region_type)
    enc_dataset = DataEncoder.run(DataEncoderParams(dataset=dataset,
                                                    encoder=cl_setting.encoder if encoder is None else encoder,
                                                    encoder_params=enc_params))

    if cl_setting.dim_reduction_method:
        enc_dataset.encoded_data.dimensionality_reduced_data = cl_setting.dim_reduction_method.fit_transform(
            enc_dataset)

    return enc_dataset
