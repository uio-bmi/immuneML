import copy
from pathlib import Path
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.generative_model.GenerativeModelLoadInstruction import GenerativeModelLoadInstruction
from immuneML.workflows.instructions.generative_model.GenerativeModelUnit import GenerativeModelUnit


class GenerativeModelLoadParser:

    """

    Specification example for GenerativeModel instruction

    Each generator requires a GenerativeModel method and path to a previously trained model. Optionally a report can be
    specified.

    DSL example for GenerativeModelInstruction assuming that m1, m2, and r1 are defined previously in
    definitions section:
    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: GenerativeModel
            generators:
                generator_1:
                    ml_method: m1
                    path: path/to/model/data
                generator_2:
                    ml_method: m2
                    path: second/path/to/model/data
                    report: r1

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> GenerativeModelLoadInstruction:
        gen_model_units = {}

        ParameterValidator.assert_keys(instruction, ["generators", "type"], "GenerativeModelLoadParser", "GenerativeModel")

        for generator_key, generator in instruction["generators"].items():
            params = self._prepare_params(generator, symbol_table, f"{key}/{generator_key}")

            gen_model_units[generator_key] = GenerativeModelUnit(**params)

        process = GenerativeModelLoadInstruction(generative_model_units=gen_model_units, name=key)
        return process

    def _prepare_params(self, generator: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:
        valid_keys = ["path", "report", "ml_method", "number_of_processes"]
        ParameterValidator.assert_keys(list(generator.keys()), valid_keys, "GenerativeModelLoadParser", "generator",
                                       False)

        params = {"path": generator["path"],
                  "report": copy.deepcopy(symbol_table.get(generator["report"])),
                  "genModel": symbol_table.get(generator["ml_method"])}

        return params


