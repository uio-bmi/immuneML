import os
import shutil

import yaml
from pathlib import Path

from source.api.galaxy.GalaxyTool import GalaxyTool
from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction


class GalaxyMLApplicationTool(GalaxyTool):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxyMLApplicationTool.__name__)
        super().__init__(specification_path, result_path, **kwargs)

    def _run(self):
        PathBuilder.build(self.result_path)
        self._check_specs()
        state = ImmuneMLApp(self.yaml_path, self.result_path).run()[0]
        shutil.copytree(list(list(state.paths.values())[0].values())[0], self.result_path / "result/")
        print("Applied ML model to the dataset, predictions are available.")

    def _check_specs(self):
        with open(self.yaml_path, "r") as file:
            specs = yaml.load(file)

        instruction_name = Util.check_instruction_type(specs, GalaxyMLApplicationTool.__name__, MLApplicationInstruction.__name__[:-11])

        ParameterValidator.assert_keys_present(list(specs['instruction'][instruction_name].keys()), ["dataset", "config_path", "label"],
                                               GalaxyMLApplicationTool.__name__, instruction_name)

        assert os.path.isfile(specs['instruction'][instruction_name]['config_path']), \
            f"{GalaxyMLApplicationTool.__name__}: file specified under 'config_path' parameter " \
            f"({specs['instruction'][instruction_name]['config_path']}) is not available. Please check if it was correctly uploaded or if the file" \
            f" name is correct."
