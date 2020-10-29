import logging
import os
import shutil
from glob import glob

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class GalaxyTrainMLModel:

    def __init__(self, specification_path, result_path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxyTrainMLModel.__name__)
        self.yaml_path = specification_path
        self.result_path = result_path if result_path[-1] == '/' else f"{result_path}/"
        self.instruction_name = None

    def run(self):
        PathBuilder.build(self.result_path)
        self._prepare_specs()
        app = ImmuneMLApp(self.yaml_path, self.result_path)
        app.run()

        model_locations = list(glob(self.result_path + f"/{self.instruction_name}/optimal_*/zip/*.zip"))

        model_export_path = PathBuilder.build(self.result_path + 'exported_models/')

        for model_location in model_locations:
            shutil.copyfile(model_location, model_export_path + os.path.basename(model_location))

        logging.info(f"{GalaxyTrainMLModel.__name__}: immuneML has finished and the trained models were exported.")

    def _prepare_specs(self):
        with open(self.yaml_path, "r") as file:
            specs = yaml.safe_load(file)

        for key in specs.keys():
            assert key in ["definitions", "instructions", "output"], f"{GalaxyTrainMLModel.__name__}: {key} is not a valid key in the YAML specification. " \
                                      f"Valid keys are: 'definitions', 'instructions'."
        ParameterValidator.assert_type_and_value(specs["instructions"], dict, GalaxyTrainMLModel.__name__, "instructions")

        assert len(list(specs["instructions"].keys())) == 1, f"{GalaxyTrainMLModel.__name__}: one instruction has to be specified under " \
                                                             f"`instructions`, got the following instead: {list(specs['instructions'].keys())}."

        self.instruction_name = list(specs["instructions"].keys())[0]

        ParameterValidator.assert_type_and_value(specs['instructions'][self.instruction_name], dict, GalaxyTrainMLModel.__name__, self.instruction_name)
        ParameterValidator.assert_keys_present(specs['instructions'][self.instruction_name].keys(), ['type'], GalaxyTrainMLModel.__name__, self.instruction_name)

        assert specs['instructions'][self.instruction_name]['type'] == TrainMLModelInstruction.__name__[:-11], \
            f"{GalaxyTrainMLModel.__name__}: instruction `type` under {self.instruction_name} has to be {TrainMLModelInstruction.__name__[:-11]} " \
            f"for this tool."

        Util.check_paths(specs, GalaxyTrainMLModel.__name__)
        Util.update_result_paths(specs, self.result_path, self.yaml_path)
