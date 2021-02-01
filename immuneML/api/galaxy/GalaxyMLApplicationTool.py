import os
import shutil
from pathlib import Path

import yaml

from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.ml_model_application.MLApplicationInstruction import MLApplicationInstruction


class GalaxyMLApplicationTool(GalaxyTool):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxyMLApplicationTool.__name__)
        super().__init__(specification_path, result_path, **kwargs)

    def _run(self):
        PathBuilder.build(self.result_path)
        self._check_specs()
        state = ImmuneMLApp(self.yaml_path, self.result_path).run()[0]
        if os.path.relpath(state.predictions_path) != os.path.relpath(self.result_path / "predictions.csv"):
            shutil.copy(state.predictions_path, self.result_path / "predictions.csv")
        print("Applied ML model to the dataset, predictions are available.")

    def _check_specs(self):
        with open(self.yaml_path, "r") as file:
            specs = yaml.load(file)

        instruction_name = Util.check_instruction_type(specs, GalaxyMLApplicationTool.__name__, MLApplicationInstruction.__name__[:-11])

        ParameterValidator.assert_keys_present(list(specs['instructions'][instruction_name].keys()), ["dataset", "config_path"],
                                               GalaxyMLApplicationTool.__name__, instruction_name)

        assert os.path.isfile(specs['instructions'][instruction_name]['config_path']), \
            f"{GalaxyMLApplicationTool.__name__}: file specified under 'config_path' parameter " \
            f"({specs['instructions'][instruction_name]['config_path']}) is not available. Please check if it was correctly uploaded or if the file" \
            f" name is correct."
