import os
import shutil
from typing import Tuple

from source.IO.ml_method.MLImport import MLImport
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.hyperparameter_optimization.HPSetting import HPSetting
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction


class MLApplicationParser:
    """
    Specification example for the MLApplication instruction:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: MLApplication
            dataset: d1
            config_path: ./config.zip
            pool_size: 1000
            label: CD
            store_encoded_data: True

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: str) -> MLApplicationInstruction:
        location = MLApplicationParser.__name__
        ParameterValidator.assert_keys(instruction.keys(), ['type', 'dataset', 'label', 'pool_size', 'config_path', 'store_encoded_data'], location, key)
        ParameterValidator.assert_in_valid_list(instruction['dataset'], symbol_table.get_keys_by_type(SymbolType.DATASET), location, f"{key}: dataset")
        ParameterValidator.assert_type_and_value(instruction['pool_size'], int, location, f"{key}: pool_size", min_inclusive=1)
        ParameterValidator.assert_type_and_value(instruction['label'], str, location, f'{key}: label')
        ParameterValidator.assert_type_and_value(instruction['config_path'], str, location, f'{key}: config_path')
        ParameterValidator.assert_type_and_value(instruction['store_encoded_data'], bool, location, f'{key}: store_encoded_data')

        hp_setting, label = self._parse_hp_setting(instruction, path, key)

        instruction = MLApplicationInstruction(dataset=symbol_table.get(instruction['dataset']), name=key, pool_size=instruction['pool_size'],
                                               label_configuration=LabelConfiguration([label]), hp_setting=hp_setting,
                                               store_encoded_data=instruction['store_encoded_data'])

        return instruction

    def _parse_hp_setting(self, instruction: dict, path: str, key: str) -> Tuple[HPSetting, Label]:

        assert os.path.isfile(instruction['config_path']), f'MLApplicationParser: {instruction["config_path"]} is not file path.'
        assert '.zip' in instruction['config_path'], f'MLApplicationParser: {instruction["config_path"]} is not a zip file.'

        config_dir = PathBuilder.build(f"{path}/unpacked_{key}/")
        shutil.unpack_archive(instruction['config_path'], config_dir, 'zip')

        hp_setting, label = MLImport.import_hp_setting(config_dir)

        return hp_setting, label
