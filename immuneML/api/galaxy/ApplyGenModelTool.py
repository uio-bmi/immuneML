import os
from pathlib import Path

import yaml

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.util.Logger import print_log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.apply_gen_model.ApplyGenModelInstruction import ApplyGenModelInstruction


class ApplyGenModelTool(GalaxyTool):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, ApplyGenModelTool.__name__)
        super().__init__(specification_path, result_path, **kwargs)

    def _run(self):
        PathBuilder.build(self.result_path)
        self._check_specs()
        state = ImmuneMLApp(self.yaml_path, self.result_path).run()[0]

        Util.export_galaxy_dataset(state.generated_dataset, self.result_path)

        print_log("Run the generative model, the resulting dataset is exported.")


    def _check_specs(self):
        with open(self.yaml_path, "r") as file:
            specs = yaml.safe_load(file)

        instruction_name = Util.check_instruction_type(specs, ApplyGenModelTool.__name__,
                                                       ApplyGenModelInstruction.__name__[:-11])

        ParameterValidator.assert_keys_present(list(specs['instructions'][instruction_name].keys()),
                                               ["ml_config_path", 'gen_examples_count'],
                                               ApplyGenModelTool.__name__, instruction_name)

        assert os.path.isfile(specs['instructions'][instruction_name]['ml_config_path']), \
            f"{ApplyGenModelTool.__name__}: file specified under 'ml_config_path' parameter " \
            f"({specs['instructions'][instruction_name]['ml_config_path']}) is not available. Please check if it was " \
            f"correctly uploaded or if the file name is correct."
