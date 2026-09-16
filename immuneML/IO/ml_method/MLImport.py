import pickle
from pathlib import Path
from typing import List, Tuple

from immuneML.IO.ml_method.MLMethodConfiguration import MLMethodConfiguration
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.util.Util import Util
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
    def import_label(config: MLMethodConfiguration) -> Label:
        return Label(name=config.label_name, values=config.label_values, positive_class=config.label_positive_class)

    @staticmethod
    def import_hp_setting(config_dir: Path, device: str = None) -> Tuple[HPSetting, Label]:

        config = MLMethodConfiguration()
        config.load(config_dir / 'ml_config.yaml')

        ml_method = ReflectionHandler.get_class_by_name(config.ml_method, 'ml_methods/')()
        if device is not None and hasattr(ml_method, 'device'):
            ml_method.device = device
        ml_method.load(config_dir)

        encoder = MLImport.import_encoder(config, config_dir)
        preprocessing_sequence = MLImport.import_preprocessing_sequence(config, config_dir)

        label = MLImport.import_label(config)
        # some MLMethod subclasses (e.g. LogRegressionCustomPenalty) don't restore self.label in their own
        # load(), since it isn't part of what they pickle; ml_config.yaml is the authoritative source anyway
        ml_method.label = label

        if getattr(ml_method, 'class_mapping', None) is None:
            # older exports may predate class_mapping being persisted; it only depends on label.values/
            # label.positive_class (not on the training data), so it can always be reconstructed here
            ml_method.class_mapping = Util.make_class_mapping(None, label)

        return HPSetting(encoder=encoder, encoder_params=config.encoding_parameters, encoder_name=config.encoding_name,
                         ml_method=ml_method, ml_method_name=config.ml_method_name, ml_params={},
                         preproc_sequence=preprocessing_sequence, preproc_sequence_name=config.preprocessing_sequence_name), label
