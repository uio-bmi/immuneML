import os
import shutil
from pathlib import Path
from typing import Tuple

from immuneML.IO.ml_method.MLImport import MLImport
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction


class MLApplicationParser:
    """
    Specification example for the MLApplication instruction:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: MLApplication
            dataset: d1
            config_path: ./config.zip
            number_of_processes: 4
            label: CD
            store_encoded_data: True

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path) -> MLApplicationInstruction:
        location = MLApplicationParser.__name__
        ParameterValidator.assert_keys(instruction.keys(), ['type', 'dataset', 'number_of_processes', 'config_path', 'store_encoded_data'], location, key)
        ParameterValidator.assert_in_valid_list(instruction['dataset'], symbol_table.get_keys_by_type(SymbolType.DATASET), location, f"{key}: dataset")
        ParameterValidator.assert_type_and_value(instruction['number_of_processes'], int, location, f"{key}: number_of_processes", min_inclusive=1)
        ParameterValidator.assert_type_and_value(instruction['config_path'], str, location, f'{key}: config_path')
        ParameterValidator.assert_type_and_value(instruction['store_encoded_data'], bool, location, f'{key}: store_encoded_data')

        hp_setting, label = self._parse_hp_setting(instruction, path, key)

        instruction = MLApplicationInstruction(dataset=symbol_table.get(instruction['dataset']), name=key, number_of_processes=instruction['number_of_processes'],
                                               label_configuration=LabelConfiguration([label]), hp_setting=hp_setting,
                                               store_encoded_data=instruction['store_encoded_data'])

        return instruction

    def _parse_hp_setting(self, instruction: dict, path: Path, key: str) -> Tuple[HPSetting, Label]:

        assert os.path.isfile(instruction['config_path']), f'MLApplicationParser: {instruction["config_path"]} is not file path.'
        assert '.zip' in instruction['config_path'], f'MLApplicationParser: {instruction["config_path"]} is not a zip file.'

        config_dir = PathBuilder.build(path / f"unpacked_{key}/")
        shutil.unpack_archive(instruction['config_path'], config_dir, 'zip')

        hp_setting, label = MLImport.import_hp_setting(config_dir)

        return hp_setting, label
