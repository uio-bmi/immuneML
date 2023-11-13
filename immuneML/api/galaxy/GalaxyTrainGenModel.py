import logging
import shutil
from pathlib import Path

import yaml

from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import TrainGenModelInstruction


class GalaxyTrainGenModel(GalaxyTool):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxyTrainGenModel.__name__)
        super().__init__(specification_path, result_path, **kwargs)
        self.instruction_name = None

    def _run(self):
        PathBuilder.build(self.result_path)
        self._prepare_specs()
        app = ImmuneMLApp(self.yaml_path, self.result_path)
        app.run()

        model_locations = list(self.result_path.glob(f"{self.instruction_name}/trained_model/*.zip"))

        model_export_path = PathBuilder.build(self.result_path / 'exported_models/')

        for model_location in model_locations:
            shutil.copyfile(model_location, model_export_path / model_location.name)

        logging.info(f"{GalaxyTrainGenModel.__name__}: immuneML has finished and the trained models were exported.")

    def _prepare_specs(self):
        with self.yaml_path.open("r") as file:
            specs = yaml.safe_load(file)

        location = GalaxyTrainGenModel.__name__

        ParameterValidator.assert_keys_present(specs.keys(), ["definitions", "instructions"], location, "YAML specification")
        ParameterValidator.assert_all_in_valid_list(specs.keys(), ["definitions", "instructions", "output"], location,
                                                    "YAML specification")

        ParameterValidator.assert_type_and_value(specs["instructions"], dict, location, "instructions")

        assert len(list(specs["instructions"].keys())) == 1, f"{location}: one instruction has to be specified under " \
                                                             f"`instructions`, got the following instead: {list(specs['instructions'].keys())}."

        self.instruction_name = list(specs["instructions"].keys())[0]

        ParameterValidator.assert_type_and_value(specs['instructions'][self.instruction_name], dict, location,
                                                 self.instruction_name)
        ParameterValidator.assert_keys_present(specs['instructions'][self.instruction_name].keys(), ['type'], location,
                                               self.instruction_name)

        assert specs['instructions'][self.instruction_name]['type'] == TrainGenModelInstruction.__name__[:-11], \
            f"{GalaxyTrainGenModel.__name__}: instruction `type` under {self.instruction_name} has to be {TrainGenModelInstruction.__name__[:-11]} " \
            f"for this tool."

        Util.check_paths(specs, location)
        Util.update_result_paths(specs, self.result_path, self.yaml_path)
