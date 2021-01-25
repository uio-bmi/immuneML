import pickle
from pathlib import Path
from typing import List, Tuple

from immuneML.IO.ml_method.MLMethodConfiguration import MLMethodConfiguration
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.ReflectionHandler import ReflectionHandler


class MLImport:

    @staticmethod
    def import_encoder(config: MLMethodConfiguration, config_dir: Path):
        encoder_class = ReflectionHandler.get_class_by_name(config.encoding_class)
        encoder = encoder_class.load_encoder(config_dir / config.encoding_file)
        return encoder

    @staticmethod
    def import_preprocessing_sequence(config: MLMethodConfiguration, config_dir) -> List[Preprocessor]:
        file_path = config_dir / config.preprocessing_file

        if file_path.is_file():
            with file_path.open("rb") as file:
                preprocessing_sequence = pickle.load(file)
        else:
            preprocessing_sequence = []
        return preprocessing_sequence

    @staticmethod
    def import_hp_setting(config_dir: Path) -> Tuple[HPSetting, Label]:

        config = MLMethodConfiguration()
        config.load(config_dir / 'ml_config.yaml')

        ml_method = ReflectionHandler.get_class_by_name(config.ml_method, 'ml_methods/')()
        ml_method.load(config_dir)

        encoder = MLImport.import_encoder(config, config_dir)
        preprocessing_sequence = MLImport.import_preprocessing_sequence(config, config_dir)

        labels = list(config.labels_with_values.keys())
        assert len(labels) == 1, "MLImport: Multiple labels set in a single ml_config file."

        label = Label(labels[0], config.labels_with_values[labels[0]])

        return HPSetting(encoder=encoder, encoder_params=config.encoding_parameters, encoder_name=config.encoding_name,
                         ml_method=ml_method, ml_method_name=config.ml_method_name, ml_params={},
                         preproc_sequence=preprocessing_sequence, preproc_sequence_name=config.preprocessing_sequence_name), label
