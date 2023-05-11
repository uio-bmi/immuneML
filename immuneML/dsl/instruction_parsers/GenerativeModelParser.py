import copy
import os
import shutil
from typing import Tuple
from pathlib import Path

from immuneML.IO.ml_method.MLImport import MLImport
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.util.PathBuilder import PathBuilder

from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.generative_model.GenerativeModelInstruction import GenerativeModelInstruction
from immuneML.workflows.instructions.generative_model.GenerativeModelUnit import GenerativeModelUnit


class GenerativeModelParser:

    """

    Specification example for GenerativeModel instruction

    Each generator requires a GenerativeModel method, encoding, dataset, and optionally a report.

    DSL example for GenerativeModelInstruction assuming that m1, m2, d1, d2, r1, r2, e1, e2 are defined previously in definitions section:
    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: GenerativeModel
            generators:
                generator_1:
                    ml_method: m1
                    encoding: e1
                    dataset: d1
                    report: r1
                generator_2:
                    ml_method: m2
                    encoding: e2
                    dataset: d2
                    report: r2

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> GenerativeModelInstruction:
        gen_model_units = {}

        ParameterValidator.assert_keys(instruction, ["generators", "type"], "GenerativeModelParser", "GenerativeModel")

        for generator_key, generator in instruction["generators"].items():
            params = self._prepare_params(generator, symbol_table, f"{key}/{generator_key}")

            gen_model_units[generator_key] = GenerativeModelUnit(**params)

        process = GenerativeModelInstruction(generative_model_units=gen_model_units, name=key)
        return process

    def _prepare_params(self, generator: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:
        valid_keys = ["dataset", "report", "ml_method", "labels", "encoding", "number_of_processes"]
        ParameterValidator.assert_keys(list(generator.keys()), valid_keys, "GenerativeModelParser", "generator",
                                       False)

        params = {"dataset": symbol_table.get(generator["dataset"]),
                  "report": copy.deepcopy(symbol_table.get(generator["report"])),
                  "genModel": symbol_table.get(generator["ml_method"]),
                  "encoder": symbol_table.get(generator["encoding"]).build_object(
                      symbol_table.get(generator["dataset"]),
                      **symbol_table.get_config(generator["encoding"])["encoder_params"]
                  )}

        return params



