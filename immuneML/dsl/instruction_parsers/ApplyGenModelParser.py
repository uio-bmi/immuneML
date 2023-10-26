import shutil
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelInstruction


class ApplyGenModelParser:
    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable,
              path: Path = None) -> ApplyGenModelInstruction:
        location = ApplyGenModelParser.__name__
        model = symbol_table.get(instruction['method'])

        ParameterValidator.assert_keys(instruction.keys(),
                                       ['type', 'gen_examples_count', 'method', 'reports', 'config_path'],
                                       location, key)
        ParameterValidator.assert_type_and_value(instruction['gen_examples_count'], int, location,
                                                 'gen_examples_count', 0)
        ParameterValidator.assert_type_and_value(instruction['config_path'], str, location, f'{key}: config_path')
        model = self._load_model(instruction, model, path)
        valid_report_ids = symbol_table.get_keys_by_type(SymbolType.REPORT)
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_report_ids, location, 'reports')

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]

        instruction = ApplyGenModelInstruction(name=key,
                                               method=model,
                                               reports=reports,
                                               gen_examples_count=instruction['gen_examples_count'])

        return instruction

    def _load_model(self, instruction: dict, generative_model: GenerativeModel, path: Path) -> GenerativeModel:
        config_dir = PathBuilder.build(path / f"unpacked/")
        shutil.unpack_archive(instruction['config_path'], config_dir, 'zip')
        return generative_model.load_model(config_dir)
