import logging
import shutil
from pathlib import Path

import yaml

from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.TrainMLModelInstruction import TrainMLModelInstruction


class GalaxyTrainMLModel(GalaxyTool):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, GalaxyTrainMLModel.__name__)
        super().__init__(specification_path, result_path, **kwargs)
        self.instruction_name = None

    def _run(self):
        PathBuilder.build(self.result_path)
        self._prepare_specs()
        app = ImmuneMLApp(self.yaml_path, self.result_path)
        app.run()

        model_locations = list(self.result_path.glob(f"{self.instruction_name}/optimal_*/zip/*.zip"))

        model_export_path = PathBuilder.build(self.result_path / 'exported_models/')

        for model_location in model_locations:
            shutil.copyfile(model_location, model_export_path / model_location.name)

        logging.info(f"{GalaxyTrainMLModel.__name__}: immuneML has finished and the trained models were exported.")

    def _prepare_specs(self):
        with self.yaml_path.open("r") as file:
            specs = yaml.safe_load(file)

        ParameterValidator.assert_keys_present(specs.keys(), ["definitions", "instructions"], GalaxyTrainMLModel.__name__, "YAML specification")
        ParameterValidator.assert_all_in_valid_list(specs.keys(), ["definitions", "instructions", "output"], GalaxyTrainMLModel.__name__,
                                                    "YAML specification")

        ParameterValidator.assert_type_and_value(specs["instructions"], dict, GalaxyTrainMLModel.__name__, "instructions")

        assert len(list(specs["instructions"].keys())) == 1, f"{GalaxyTrainMLModel.__name__}: one instruction has to be specified under " \
                                                             f"`instructions`, got the following instead: {list(specs['instructions'].keys())}."

        self.instruction_name = list(specs["instructions"].keys())[0]

        ParameterValidator.assert_type_and_value(specs['instructions'][self.instruction_name], dict, GalaxyTrainMLModel.__name__,
                                                 self.instruction_name)
        ParameterValidator.assert_keys_present(specs['instructions'][self.instruction_name].keys(), ['type'], GalaxyTrainMLModel.__name__,
                                               self.instruction_name)

        assert specs['instructions'][self.instruction_name]['type'] == TrainMLModelInstruction.__name__[:-11], \
            f"{GalaxyTrainMLModel.__name__}: instruction `type` under {self.instruction_name} has to be {TrainMLModelInstruction.__name__[:-11]} " \
            f"for this tool."

        assert len(
            specs['instructions'][self.instruction_name]['labels']) == 1, f"{GalaxyTrainMLModel.__name__}: one label has to be specified under " \
                                                                          f"`labels`, got the following instead: {specs['instructions'][self.instruction_name]['labels']}."
        Util.check_paths(specs, GalaxyTrainMLModel.__name__)
        Util.update_result_paths(specs, self.result_path, self.yaml_path)
