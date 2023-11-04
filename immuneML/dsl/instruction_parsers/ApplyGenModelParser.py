import os
import shutil
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelInstruction


class ApplyGenModelParser:
    """
    Specification example for the ApplyGenModel instruction:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: ApplyGenModel
            gen_examples_count: 100
            method: m1
            config_path: ./config.zip
            reports: [data_rep1, ml_rep2]
    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable,
              path: Path = None) -> ApplyGenModelInstruction:
        location = ApplyGenModelParser.__name__

        ParameterValidator.assert_keys(instruction.keys(),
                                       ['type', 'gen_examples_count', 'method', 'reports', 'config_path'],
                                       location, key)
        ParameterValidator.assert_type_and_value(instruction['type'], str, location, 'type')
        ParameterValidator.assert_type_and_value(instruction['gen_examples_count'], int, location,
                                                 'gen_examples_count', 0)
        ParameterValidator.assert_type_and_value(instruction['method'], str, location, 'method')
        valid_report_ids = symbol_table.get_keys_by_type(SymbolType.REPORT)
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_report_ids, location, 'reports')
        ParameterValidator.assert_type_and_value(instruction['config_path'], str, location, f'{key}: config_path')

        base_model = symbol_table.get(instruction['method'])
        read_model = self._load_model(instruction['config_path'], key, base_model, path)

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]

        instruction = ApplyGenModelInstruction(name=key,
                                               gen_examples_count=instruction['gen_examples_count'],
                                               method=read_model,
                                               reports=reports)

        return instruction

    def _load_model(self, model_path: str, instruction_key: str, base_model: GenerativeModel,
                    path: Path) -> GenerativeModel:
        assert os.path.isfile(model_path), f'{ApplyGenModelParser.__name__}: {model_path} is not file path.'
        assert '.zip' in model_path, f'{ApplyGenModelParser.__name__}: {model_path} is not a zip file.'

        config_dir = PathBuilder.build(path / f"unpacked/{instruction_key}")
        shutil.unpack_archive(model_path, config_dir, 'zip')

        return base_model.load_model(config_dir)
