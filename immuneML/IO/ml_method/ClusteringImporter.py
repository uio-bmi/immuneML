import pickle
from pathlib import Path
from typing import Tuple

import yaml

from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem, ClusteringSetting


class ClusteringImporter:
    """Import clustering settings and items from exported zip files."""

    @staticmethod
    def import_clustering_item(config_dir: Path) -> Tuple[ClusteringItem, dict]:
        """
        Import a ClusteringItem from an exported directory.

        Args:
            config_dir: Path to the directory containing the exported clustering files

        Returns:
            Tuple of (ClusteringItem, config dict)
        """
        config = ClusteringImporter._load_config(config_dir)

        encoder = ClusteringImporter._import_encoder(config, config_dir)
        clustering_method = ClusteringImporter._import_clustering_method(config, config_dir)
        dim_reduction = ClusteringImporter._import_dim_reduction(config, config_dir)
        classifier = ClusteringImporter._import_classifier(config, config_dir)

        cl_setting = ClusteringSetting(
            encoder=encoder,
            encoder_params=config.get('encoder_params', {}),
            encoder_name=config.get('encoder_name', ''),
            clustering_method=clustering_method,
            clustering_params=config.get('clustering_params', {}),
            clustering_method_name=config.get('clustering_method_name', ''),
            dim_reduction_method=dim_reduction,
            dim_red_params=config.get('dim_reduction_params'),
            dim_red_name=config.get('dim_reduction_name')
        )

        cl_item = ClusteringItem(
            encoder=encoder,
            method=clustering_method,
            cl_setting=cl_setting,
            classifier=classifier
        )

        return cl_item, config

    @staticmethod
    def _load_config(config_dir: Path) -> dict:
        """Load the clustering configuration YAML file."""
        config_path = config_dir / 'clustering_config.yaml'
        with config_path.open('r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _import_encoder(config: dict, config_dir: Path):
        """Import the encoder from the exported files."""
        encoder_class = ReflectionHandler.get_class_by_name(config['encoding_class'], 'encodings/')
        encoder = encoder_class.load_encoder(config_dir / config['encoding_file'])
        return encoder

    @staticmethod
    def _import_clustering_method(config: dict, config_dir: Path):
        """Import the clustering method from pickle."""
        method_path = config_dir / config['clustering_method_file']
        with method_path.open('rb') as f:
            return pickle.load(f)

    @staticmethod
    def _import_dim_reduction(config: dict, config_dir: Path):
        """Import the dimensionality reduction method if present."""
        if config.get('dim_reduction_file'):
            dim_red_path = config_dir / config['dim_reduction_file']
            with dim_red_path.open('rb') as f:
                return pickle.load(f)
        return None

    @staticmethod
    def _import_classifier(config: dict, config_dir: Path):
        """Import the sklearn classifier used for result-based validation."""
        if config.get('classifier_filename'):
            classifier_path = config_dir / config['classifier_filename']
            with classifier_path.open('rb') as f:
                return pickle.load(f)
        return None