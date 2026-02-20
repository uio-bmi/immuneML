import os
import pickle
import shutil
from pathlib import Path
from typing import List

from immuneML.IO.ml_method.MLMethodConfiguration import MLMethodConfiguration
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.PathBuilder import PathBuilder


class MLExporter:

    @staticmethod
    def export_zip(hp_item: HPItem, path: Path, label_name: str) -> str:
        state_path = path.absolute()
        export_path = MLExporter.export(hp_item, state_path / "exported")
        filename = f"ml_settings_{label_name}"
        abs_zip_path = Path(shutil.make_archive(state_path / "zip" / filename, "zip", str(export_path))).absolute()
        return abs_zip_path

    @staticmethod
    def export(hp_item: HPItem, path: Path) -> Path:
        PathBuilder.build(path)
        preproc_filename = MLExporter._store_preprocessing_sequence(hp_item.hp_setting.preproc_sequence, path).name
        encoder_filename = MLExporter._store_encoder(hp_item.hp_setting.encoder, path).name

        MLExporter.store_ml_method(hp_item.method, path, preproc_filename, encoder_filename,
                                   hp_item.hp_setting.encoder_name, hp_item.hp_setting.encoder_params,
                                   type(hp_item.hp_setting.encoder).__name__, hp_item.train_dataset,
                                   hp_item.hp_setting.preproc_sequence_name, hp_item.hp_setting.preproc_sequence)

        return path

    @staticmethod
    def store_ml_method(method: MLMethod, path: Path, preproc_filename, encoder_filename, encoder_name,
                        encoder_params, encoder_class_name, train_dataset, preproc_sequence_name,
                        preproc_sequence):
        method.store(path)

        method_config = MLMethodConfiguration(label_name=method.get_label_name(),
                                              label_positive_class=method.get_positive_class(),
                                              label_values=method.get_classes(),
                                              software_used=method.get_package_info(),
                                              encoding_name=encoder_name,
                                              encoding_parameters=encoder_params,
                                              encoding_file=encoder_filename,
                                              encoding_class=encoder_class_name,
                                              ml_method=type(method).__name__,
                                              ml_method_name=method.name,
                                              train_dataset_id=train_dataset.identifier,
                                              train_dataset_name=train_dataset.name,
                                              preprocessing_sequence_name=preproc_sequence_name,
                                              preprocessing_file=os.path.basename(preproc_filename),
                                              preprocessing_parameters={
                                                  type(seq).__name__: {str(key): str(val) for key, val in
                                                                       vars(seq).items()}
                                                  for seq in preproc_sequence})

        method_config.store(path / 'ml_config.yaml')

    @staticmethod
    def _store_encoder(encoder, path: Path) -> Path:
        filename = path / "encoder.pickle"
        type(encoder).store_encoder(encoder, filename)
        return filename

    @staticmethod
    def _store_preprocessing_sequence(preprocessing_sequence: List[Preprocessor], path: Path) -> Path:
        filename = path / "preprocessing_sequence.pickle"

        with filename.open("wb") as file:
            pickle.dump(preprocessing_sequence, file)

        return filename
