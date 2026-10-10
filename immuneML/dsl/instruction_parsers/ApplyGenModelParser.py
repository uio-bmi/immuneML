import os
import shutil
from pathlib import Path

from immuneML.data_model.bnp_util import read_yaml
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelInstruction


class ApplyGenModelParser:
    """
    Specification example for the ApplyGenModel instruction:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: ApplyGenModel
            gen_examples_count: 100
            ml_config_path: ./config.zip
            reports: [data_rep1, ml_rep2]
            p_gen_datasets: [my_dataset1, my_dataset2] # optional
    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable,  path: Path = None) -> ApplyGenModelInstruction:
        location = ApplyGenModelParser.__name__

        required_keys = ['type', 'gen_examples_count', 'reports', 'ml_config_path']
        ParameterValidator.assert_keys(instruction.keys(), required_keys + ['p_gen_datasets'], location, key, exclusive=False)
        ParameterValidator.assert_keys_present(list(instruction.keys()), required_keys, location, key)
        ParameterValidator.assert_type_and_value(instruction['type'], str, location, 'type')
        ParameterValidator.assert_type_and_value(instruction['gen_examples_count'], int, location,
                                                 'gen_examples_count', 0)
        assert instruction['gen_examples_count'] > 0 or instruction.get('p_gen_datasets'), \
            (f"{location}: gen_examples_count is 0 and p_gen_datasets is not specified under {key}, so there is nothing to "
             f"do. Set gen_examples_count to at least 1 or specify p_gen_datasets to compute generation probabilities.")
        valid_report_ids = symbol_table.get_keys_by_type(SymbolType.REPORT)
        ParameterValidator.assert_all_in_valid_list(instruction['reports'], valid_report_ids, location, 'reports')
        ParameterValidator.assert_type_and_value(instruction['ml_config_path'], str, location, f'{key}: ml_config_path')

        gen_model = self._load_model(instruction['ml_config_path'], key, path)

        reports = [symbol_table.get(report_id) for report_id in instruction['reports']]
        p_gen_datasets = self._parse_p_gen_datasets(instruction, symbol_table, gen_model, key)

        instruction = ApplyGenModelInstruction(name=key,
                                               gen_examples_count=instruction['gen_examples_count'],
                                               method=gen_model,
                                               reports=reports,
                                               p_gen_datasets=p_gen_datasets)

        return instruction

    def _parse_p_gen_datasets(self, instruction: dict, symbol_table: SymbolTable, gen_model: GenerativeModel,
                              key: str) -> dict:
        location = ApplyGenModelParser.__name__

        if instruction.get('p_gen_datasets') is None:
            return {}

        ParameterValidator.assert_type_and_value(instruction['p_gen_datasets'], list, location,
                                                 f'{key}: p_gen_datasets')
        ParameterValidator.assert_all_in_valid_list(instruction['p_gen_datasets'],
                                                    symbol_table.get_keys_by_type(SymbolType.DATASET), location,
                                                    f'{key}: p_gen_datasets')
        datasets = {name: symbol_table.get(name) for name in instruction['p_gen_datasets']}

        for name, dataset in datasets.items():
            assert isinstance(dataset, SequenceDataset), \
                (f"{location}: generation probabilities can for now be computed only for sequence datasets, but "
                 f"{name} is a {type(dataset).__name__}.")

        assert len(datasets) == 0 or gen_model.can_compute_p_gens(), \
            (f"{location}: {type(gen_model).__name__} cannot compute generation probabilities, so 'p_gen_datasets' "
             f"cannot be specified under {key}.")

        return datasets

    def _load_model(self, model_path: str, instruction_key: str, path: Path) -> GenerativeModel:
        location = ApplyGenModelParser.__name__
        assert os.path.isfile(model_path), f'{location}: {model_path} is not file path.'
        assert '.zip' in model_path, f'{location}: {model_path} is not a zip file.'

        config_dir = PathBuilder.build(path / f"unpacked/{instruction_key}")
        shutil.unpack_archive(model_path, config_dir, 'zip')

        model_overview = read_yaml(config_dir / 'model_overview.yaml')
        assert isinstance(model_overview, dict) and 'type' in model_overview and isinstance(model_overview['type'], str), \
            f"{location}: invalid format of model_overview.yaml from the zip file."

        valid_gen_models = ReflectionHandler.all_nonabstract_subclass_basic_names(GenerativeModel, "", "ml_methods/generative_models/")
        ParameterValidator.assert_in_valid_list(model_overview['type'], valid_gen_models, location, 'type')

        gen_model_class = ReflectionHandler.get_class_by_name(model_overview['type'], 'ml_methods/generative_models/')
        return gen_model_class.load_model(config_dir)
