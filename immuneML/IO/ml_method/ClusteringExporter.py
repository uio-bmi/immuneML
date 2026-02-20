import pickle
import shutil
from pathlib import Path

import yaml

from immuneML.IO.ml_method.MLExporter import MLExporter
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem, ClusteringSetting


class ClusteringExporter:

    @staticmethod
    def export_zip(cl_item: ClusteringItem, path: Path, setting_key: str) -> Path:
        """
        Export a clustering setting (encoder, dim reduction if any, clustering method) as a zip file.

        Args:
            cl_item: The ClusteringItem containing the fitted encoder and clustering method
            path: The directory where the export will be stored
            setting_key: A key identifying this setting (used for naming)

        Returns:
            Path to the created zip file
        """
        state_path = path.absolute()
        export_path = ClusteringExporter.export(cl_item, state_path / "fitted")
        abs_zip_path = Path(shutil.make_archive(str(state_path / setting_key), "zip", str(export_path))).absolute()
        return abs_zip_path

    @staticmethod
    def export(cl_item: ClusteringItem, path: Path) -> Path:
        """
        Export a clustering item's components to disk.

        Args:
            cl_item: The ClusteringItem to export
            path: The directory where files will be stored

        Returns:
            Path to the export directory
        """
        PathBuilder.build(path)

        # Store encoder
        encoder_filename = ClusteringExporter._store_encoder(cl_item.encoder, path).name

        # Store clustering method
        method_filename = ClusteringExporter._store_clustering_method(cl_item.method, path).name

        # Store classifier (sklearn object for result-based validation)
        classifier_filename = None
        if cl_item.classifier is not None:
            classifier_filename = ClusteringExporter._store_classifier(cl_item.classifier, path).name

        # Store dim reduction method if present (use the fitted one from cl_item)
        dim_red_filename = None
        if cl_item.dim_red_method is not None:
            dim_red_filename = ClusteringExporter._store_dim_reduction(
                cl_item.dim_red_method, path
            ).name

        # Create configuration
        config = ClusteringExporter._create_config(cl_item, encoder_filename, method_filename, dim_red_filename,
                                                   classifier_filename)
        config_path = path / 'clustering_config.yaml'
        with config_path.open('w') as f:
            yaml.dump(config, f)

        return path

    @staticmethod
    def _store_encoder(encoder: DatasetEncoder, path: Path) -> Path:
        filename = path / "encoder.pickle"
        type(encoder).store_encoder(encoder, filename)
        return filename

    @staticmethod
    def _store_clustering_method(method, path: Path) -> Path:
        filename = path / "clustering_method.pickle"
        with filename.open("wb") as file:
            pickle.dump(method, file)
        return filename

    @staticmethod
    def _store_dim_reduction(dim_red_method, path: Path) -> Path:
        filename = path / "dim_reduction.pickle"
        with filename.open("wb") as file:
            pickle.dump(dim_red_method, file)
        return filename

    @staticmethod
    def _store_classifier(classifier, path: Path) -> Path:
        """Store the sklearn classifier used for result-based validation."""
        filename = path / "classifier.pickle"
        with filename.open("wb") as file:
            pickle.dump(classifier, file)
        return filename

    @staticmethod
    def _create_config(cl_item: ClusteringItem, encoder_filename: str, method_filename: str,
                       dim_red_filename: str = None, classifier_filename: str = None) -> dict:
        """Create a configuration dictionary for the clustering export."""
        config = {
            'encoding_file': encoder_filename,
            'encoding_class': type(cl_item.encoder).__name__ if cl_item.encoder else None,
            'clustering_method_file': method_filename,
            'clustering_method_class': type(cl_item.method).__name__ if cl_item.method else None,
            'clustering_method_name': cl_item.method.name if cl_item.method else None,
            'classifier_filename': classifier_filename
        }

        if cl_item.cl_setting:
            config.update({
                'setting_key': cl_item.cl_setting.get_key(),
                'encoder_name': cl_item.cl_setting.encoder_name,
                'encoder_params': cl_item.cl_setting.encoder_params,
                'clustering_params': cl_item.cl_setting.clustering_params,
            })

            if cl_item.dim_red_method is not None:
                config.update({
                    'dim_reduction_file': dim_red_filename,
                    'dim_reduction_class': type(cl_item.dim_red_method).__name__,
                    'dim_reduction_name': cl_item.cl_setting.dim_red_name,
                    'dim_reduction_params': cl_item.cl_setting.dim_red_params,
                })

        if cl_item.dataset:
            config.update({
                'discovery_dataset_id': cl_item.dataset.identifier,
                'discovery_dataset_name': cl_item.dataset.name,
            })

        return config