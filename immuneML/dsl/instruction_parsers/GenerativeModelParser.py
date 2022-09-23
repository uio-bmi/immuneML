import copy
from pathlib import Path

from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.GenerativeModelInstruction import GenerativeModelInstruction
from immuneML.workflows.instructions.generative_model.GenerativeModelUnit import GenerativeModelUnit


class GenerativeModelParser:

    """

    Specification example for GenerativeModel instruction

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: LSTM
            generators:
                generator_1:
                    encoding: e1
                    dataset: d1
                    report: r1
                generator_2:
                    encoding: e2
                    dataset: d2
                    report: r2
    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> GenerativeModelInstruction:
        gen_model_units = {}

        ParameterValidator.assert_keys(instruction, ["generators", "type"], "GenerativeModelParser", "GenerativeModel")
        #ParameterValidator.assert_type_and_value(instruction["number_of_processes"], int, GenerativeModelInstruction.__name__, "number_of_processes")

        for generator_key, generator in instruction["generators"].items():

            params = self._prepare_params(generator, symbol_table, f"{key}/{generator_key}")
            #params["number_of_processes"] = instruction["number_of_processes"]
            gen_model_units[generator_key] = GenerativeModelUnit(**params)

        process = GenerativeModelInstruction(generative_model_units=gen_model_units, name=key)
        return process

    def _prepare_params(self, generator: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:
        valid_keys = ["dataset", "report", "preprocessing_sequence", "labels", "encoding", "number_of_processes"]
        ParameterValidator.assert_keys(list(generator.keys()), valid_keys, "GenerativeModelParser", "generator",
                                       False)

        params = {"dataset": symbol_table.get(generator["dataset"]),
                  "report": copy.deepcopy(symbol_table.get(generator["report"]))}

        optional_params = self._prepare_optional_params(generator, symbol_table, yaml_location)
        params = {**params, **optional_params}


    def _prepare_optional_params(self, generator: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        params = {}
        dataset = symbol_table.get(generator["dataset"])

        if "encoding" in generator:
            params["encoder"] = symbol_table.get(generator["encoding"]).build_object(dataset, **symbol_table.get_config(generator["encoding"])["encoder_params"])

        if "labels" in generator:
            params["label_config"] = LabelHelper.create_label_config(generator["labels"], dataset, GenerativeModelParser.__name__, yaml_location)
        else:
            params["label_config"] = LabelConfiguration()

        if "preprocessing_sequence" in generator:
            params["preprocessing_sequence"] = symbol_table.get(generator["preprocessing_sequence"])

        return params


