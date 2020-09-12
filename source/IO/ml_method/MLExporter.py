import os
import pickle
from typing import List

from source.IO.ml_method.MLMethodConfiguration import MLMethodConfiguration
from source.hyperparameter_optimization.states.HPItem import HPItem
from source.preprocessing.Preprocessor import Preprocessor
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder


class MLExporter:

    @staticmethod
    def export_zip(hp_item: HPItem, path: str) -> str:
        export_path = MLExporter.export(hp_item, path + "exported/")
        filename = f"ml_model_{hp_item.hp_setting.ml_method_name}"
        model_zip_path = Util.make_downloadable_zip(path, export_path, filename)
        return model_zip_path

    @staticmethod
    def export(hp_item: HPItem, path: str) -> str:
        PathBuilder.build(path)
        preproc_filename = os.path.basename(MLExporter._store_preprocessing_sequence(hp_item.hp_setting.preproc_sequence, path))
        encoder_filename = os.path.basename(MLExporter._store_encoder(hp_item.hp_setting.encoder, path))

        hp_item.method.store(path, hp_item.method.get_feature_names())
        labels_with_values = {label: hp_item.method.get_classes_for_label(label).tolist() for label in hp_item.method.get_labels()}

        method_config = MLMethodConfiguration(labels_with_values=labels_with_values, software_used=hp_item.method.get_package_info(),
                                              encoding_name=hp_item.hp_setting.encoder_name, encoding_parameters=hp_item.hp_setting.encoder_params,
                                              encoding_file=encoder_filename, encoding_class=type(hp_item.hp_setting.encoder).__name__,
                                              ml_method=type(hp_item.method).__name__, ml_method_name=hp_item.method.name,
                                              train_dataset_id=hp_item.train_dataset.identifier, train_dataset_name=hp_item.train_dataset.name,
                                              preprocessing_sequence_name=hp_item.hp_setting.preproc_sequence_name,
                                              preprocessing_file=os.path.basename(preproc_filename),
                                              preprocessing_parameters={type(seq).__name__: vars(seq) for seq in hp_item.hp_setting.preproc_sequence})

        method_config.store(path + 'ml_config.yaml')

        return path

    @staticmethod
    def _store_encoder(encoder, path: str) -> str:
        filename = f"{path}encoder.pickle"
        type(encoder).store_encoder(encoder, filename)
        return filename

    @staticmethod
    def _store_preprocessing_sequence(preprocessing_sequence: List[Preprocessor], path: str) -> str:
        filename = f"{path}preprocessing_sequence.pickle"

        with open(filename, "wb") as file:
            pickle.dump(preprocessing_sequence, file)

        return filename
