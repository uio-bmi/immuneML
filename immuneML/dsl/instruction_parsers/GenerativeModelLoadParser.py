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
from immuneML.workflows.instructions.generative_model.GenerativeModelLoadInstruction import GenerativeModelLoadInstruction
from immuneML.workflows.instructions.generative_model.GenerativeModelUnit import GenerativeModelUnit


class GenerativeModelLoadParser:

    """

    Specification example for GenerativeModel instruction

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: GenerativeModelLoad
            generators:
                generator_1:
                    report: r1
                    ml_method: LSTM
                generator_2:
                    report: r2
                    ml_method: PWM
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
        valid_keys = ["path", "report", "ml_method", "number_of_processes", "amount"]
        ParameterValidator.assert_keys(list(generator.keys()), valid_keys, "GenerativeModelLoadParser", "generator",
                                       False)

        params = {"path": generator["path"],
                  "report": copy.deepcopy(symbol_table.get(generator["report"])),
                  "genModel": symbol_table.get(generator["ml_method"])}

        optional_params = self._prepare_optional_params(generator, symbol_table, yaml_location)
        params = {**params, **optional_params}

        return params

    def _prepare_optional_params(self, generator: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        params = {}
        if "amount" in generator:
            params["amount"] = generator["amount"]

        return params


