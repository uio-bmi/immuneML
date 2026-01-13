import copy
import logging
import numbers
from pathlib import Path
from typing import Tuple, Dict, List, Union

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import issparse
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.ml_methods.helper_methods.FurthestNeighborClassifier import FurthestNeighborClassifier
from immuneML.ml_metrics.ClusteringMetric import is_external, is_internal
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem, DataFrameWrapper, \
    ClusteringSetting
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringItemResult
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


def run_all_settings(dataset: Dataset, clustering_settings: List[ClusteringSetting], path: Path,
                     predictions_df: pd.DataFrame, metrics: List[str], label_config: LabelConfiguration,
                     number_of_processes: int, sequence_type: SequenceType, region_type: RegionType,
                     report_handler=None, run_id: int = None, state=None) -> Tuple[Dict, pd.DataFrame]:
    """
    Run all clustering settings on a dataset and collect results.

    Args:
        dataset: The dataset to cluster
        clustering_settings: List of clustering settings to evaluate
        path: Output path for results
        predictions_df: DataFrame to store predictions
        metrics: List of metric names to compute
        label_config: Label configuration for external metrics
        number_of_processes: Number of processes for parallelization
        sequence_type: Sequence type for encoding
        region_type: Region type for encoding
        report_handler: Optional report handler for running item reports
        run_id: Optional run identifier
        state: Optional clustering state for report handler

    Returns:
        Tuple of (clustering_items dict, updated predictions_df)
    """

    clustering_items = {}

    for cl_setting in clustering_settings:
        cl_item, predictions_df = run_setting(
            dataset=dataset,
            cl_setting=cl_setting,
            path=path,
            predictions_df=predictions_df,
            metrics=metrics,
            label_config=label_config,
            number_of_processes=number_of_processes,
            sequence_type=sequence_type,
            region_type=region_type,
            report_handler=report_handler,
            run_id=run_id,
            state=state
        )
        clustering_items[cl_setting.get_key()] = cl_item

    return clustering_items, predictions_df


def run_setting(dataset: Dataset, cl_setting: ClusteringSetting, path: Path,
                predictions_df: pd.DataFrame, metrics: List[str], label_config: LabelConfiguration,
                number_of_processes: int, sequence_type: SequenceType, region_type: RegionType,
                report_handler=None, run_id: int = None, state=None, evaluate: bool = True) \
        -> Tuple[ClusteringItemResult, pd.DataFrame]:
    """
    Run a single clustering setting on a dataset.

    Args:
        dataset: The dataset to cluster
        cl_setting: The clustering setting to use
        path: Output path for results
        predictions_df: DataFrame to store predictions
        metrics: List of metric names to compute
        label_config: Label configuration for external metrics
        number_of_processes: Number of processes for parallelization
        sequence_type: Sequence type for encoding
        region_type: Region type for encoding
        report_handler: Optional report handler for running item reports
        run_id: Optional run identifier
        state: Optional clustering state for report handler
        evaluate: Whether to compute internal/external evaluation metrics

    Returns:
        Tuple of (ClusteringItemResult, updated predictions_df)
    """
    print_log(f"Running clustering setting {cl_setting.get_key()}")

    cl_setting.path = PathBuilder.build(path / f"{cl_setting.get_key()}")

    # Encode data
    encoder = copy.deepcopy(cl_setting.encoder)
    enc_dataset = encode_dataset(dataset, cl_setting, number_of_processes, label_config,
                                 True, sequence_type, region_type, encoder)

    # Run clustering
    method = copy.deepcopy(cl_setting.clustering_method)
    predictions = fit_and_predict(enc_dataset, method)
    if predictions_df is not None:
        predictions_df[f'predictions_{cl_setting.get_key()}'] = predictions

    print_log(f"{cl_setting.get_key()}: clustering method fitted and predictions made.")

    cl_item = ClusteringItem(
        cl_setting=cl_setting,
        dataset=enc_dataset,
        predictions=predictions,
        encoder=encoder,
        method=method
    )

    if evaluate:
        cl_item = evaluate_clustering(predictions_df, cl_setting, get_features(enc_dataset, cl_setting), metrics, label_config, cl_item)

    # Run reports if handler provided
    report_results = []
    if report_handler is not None and run_id is not None and state is not None:
        report_results = report_handler.run_item_reports(cl_item, run_id, cl_setting.path, state)

    print_log(f"Clustering setting {cl_setting.get_key()} finished.")

    return ClusteringItemResult(cl_item, report_results), predictions_df


def fit_and_predict(dataset: Dataset, method: ClusteringMethod) -> np.ndarray:
    """Fit clustering method and get predictions."""
    if hasattr(method, 'fit_predict'):
        return method.fit_predict(dataset)
    else:
        method.fit(dataset)
        return method.predict(dataset)


def get_features(dataset: Dataset, cl_setting: ClusteringSetting):
    """Get features from encoded dataset."""
    return dataset.encoded_data.examples if cl_setting.dim_reduction_method is None \
        else dataset.encoded_data.dimensionality_reduced_data


def evaluate_clustering(predictions_df: pd.DataFrame, cl_setting: ClusteringSetting,
                        features, metrics: List[str], label_config: LabelConfiguration,
                        cl_item: ClusteringItem) -> ClusteringItem:
    """
    Evaluate clustering results using internal and external metrics.

    Args:
        predictions_df: DataFrame containing predictions and labels
        cl_setting: The clustering setting used
        features: Feature matrix for internal metrics
        metrics: List of metric names to compute
        label_config: Label configuration for external metrics
        cl_item: Clustering item to evaluate and update with performance csv files

    Returns:
        Updated ClusteringItem with performance CSV file paths
    """

    cl_item.external_performance = DataFrameWrapper(path=eval_external_metrics(predictions_df, cl_setting, metrics, label_config))
    cl_item.internal_performance = DataFrameWrapper(path=eval_internal_metrics(predictions_df, cl_setting, features, metrics))

    return cl_item


def eval_internal_metrics(predictions_df: pd.DataFrame, cl_setting: ClusteringSetting,
                          features, metrics: List[str]) -> Path:
    """
    Evaluate internal clustering metrics (based on features and predictions only).

    Args:
        predictions_df: DataFrame containing predictions
        cl_setting: The clustering setting used
        features: Feature matrix
        metrics: List of metric names to compute

    Returns:
        Path to the internal performances CSV file
    """
    internal_performances = {}
    dense_features = features.toarray() if issparse(features) else features

    for metric in [m for m in metrics if is_internal(m)]:
        try:
            metric_fn = getattr(sklearn.metrics, metric)
            internal_performances[metric] = [metric_fn(dense_features,
                                                       predictions_df[f'predictions_{cl_setting.get_key()}'].values)]
        except ValueError as e:
            logging.warning(f"Error calculating metric {metric}: {e}")
            internal_performances[metric] = [np.nan]

    pd.DataFrame(internal_performances).to_csv(str(cl_setting.path / 'internal_performances.csv'), index=False)
    return cl_setting.path / 'internal_performances.csv'


def eval_external_metrics(predictions_df: pd.DataFrame, cl_setting: ClusteringSetting,
                          metrics: List[str], label_config: LabelConfiguration) -> Union[Path, None]:
    """
    Evaluate external clustering metrics (comparing predictions to ground truth labels).

    Args:
        predictions_df: DataFrame containing predictions and labels
        cl_setting: The clustering setting used
        metrics: List of metric names to compute
        label_config: Label configuration with label names

    Returns:
        Path to the external performances CSV file, or None if no labels
    """
    if label_config is not None and label_config.get_label_count() > 0:
        external_performances = {label: {} for label in label_config.get_labels_by_name()}

        for metric in [m for m in metrics if is_external(m)]:
            metric_fn = getattr(sklearn.metrics, metric)
            for label in label_config.get_labels_by_name():
                try:
                    external_performances[label][metric] = metric_fn(
                        predictions_df[label].values,
                        predictions_df[f'predictions_{cl_setting.get_key()}'].values
                    )
                except ValueError as e:
                    logging.warning(f"Error calculating metric {metric}: {e}")
                    external_performances[label][metric] = np.nan

        (pd.DataFrame(external_performances).reset_index().rename(columns={'index': 'metric'})
         .to_csv(str(cl_setting.path / 'external_performances.csv'), index=False))

        return cl_setting.path / 'external_performances.csv'
    else:
        return None


def train_cluster_classifier(cl_item: ClusteringItem):
    classifier = get_complementary_classifier(cl_item.cl_setting)

    # Get features and cluster assignments from discovery data
    features = get_features(cl_item.dataset, cl_item.cl_setting)

    if len(list(set(cl_item.predictions))) == 1:
        return cl_item.predictions[0]
    else:
        classifier.fit(features, cl_item.predictions)
        return classifier


def apply_cluster_classifier(dataset: Dataset, cl_setting: ClusteringSetting, classifier, encoder: DatasetEncoder,
                             predictions_path: Path, number_of_processes: int, sequence_type: SequenceType,
                             region_type: RegionType) -> ClusteringItem:
    """Apply trained classifier to validation data."""

    enc_dataset = encode_dataset(dataset, cl_setting, number_of_processes, LabelConfiguration(),
                                 learn_model=False, sequence_type=sequence_type,
                                 region_type=region_type, encoder=encoder)
    features = get_features(enc_dataset, cl_setting)

    if isinstance(classifier, numbers.Number):
        predictions = [classifier] * dataset.get_example_count()
        logging.warning(f"Only one cluster found in discovery data. Assigning all validation data to "
                        f"cluster {classifier}.")
    else:
        predictions = classifier.predict(features)

    cl_item = ClusteringItem(
        cl_setting=cl_setting,
        dataset=enc_dataset,
        predictions=predictions
    )

    pd.DataFrame({'predictions': predictions, 'example_id': dataset.get_example_ids()}).to_csv(
        predictions_path, index=False)

    return cl_item


def encode_dataset(dataset: Dataset, cl_setting: ClusteringSetting, number_of_processes: int,
                   label_config: LabelConfiguration, learn_model: bool, sequence_type: SequenceType,
                   region_type: RegionType, encoder: DatasetEncoder = None):
    """
    Encode a dataset using the specified clustering setting's encoder.
    Results are cached based on parameters.

    Args:
        dataset: The dataset to encode
        cl_setting: The clustering setting containing encoder configuration
        number_of_processes: Number of processes for parallelization
        label_config: Label configuration
        learn_model: Whether to learn the encoder model or use existing
        sequence_type: Sequence type for encoding
        region_type: Region type for encoding
        encoder: Optional pre-configured encoder

    Returns:
        Encoded dataset
    """
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
    """
    Internal function to encode a dataset (called by encode_dataset with caching).

    Args:
        dataset: The dataset to encode
        cl_setting: The clustering setting containing encoder configuration
        number_of_processes: Number of processes for parallelization
        label_config: Label configuration
        learn_model: Whether to learn the encoder model or use existing
        sequence_type: Sequence type for encoding
        region_type: Region type for encoding
        encoder: Optional pre-configured encoder

    Returns:
        Encoded dataset with optional dimensionality reduction
    """
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
        enc_dataset.encoded_data.dim_names = cl_setting.dim_reduction_method.get_dimension_names()

    return enc_dataset

def get_complementary_classifier(cl_setting: ClusteringSetting):
    """
    Returns a complementary classifier based on the clustering method.

    Args:
        cl_setting: ClusteringSetting object containing the clustering method configuration

    Returns:
        An instance of the appropriate classifier; kNN if no matches are found
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

    return KNeighborsClassifier()