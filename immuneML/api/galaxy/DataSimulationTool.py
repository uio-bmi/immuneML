import logging
import shutil
from pathlib import Path

import yaml

from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.dataset_generation.DatasetExportInstruction import DatasetExportInstruction


class DataSimulationTool(GalaxyTool):

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, DataSimulationTool.__name__)
        super().__init__(specification_path, result_path, **kwargs)
        self.expected_instruction = DatasetExportInstruction.__name__[:-11]
        self.instruction_name = None
        self.dataset_name = None
        self.export_format = None

    def _run(self):
        self.prepare_specs()
        Util.run_tool(self.yaml_path, self.result_path)

        dataset_location = self.result_path / f"{self.instruction_name}/{self.dataset_name}/{self.export_format}/"
        shutil.copytree(dataset_location, self.result_path / 'result/')

        logging.info(f"{DataSimulationTool.__name__}: immuneML has finished and the dataset was created.")

    def prepare_specs(self):
        with self.yaml_path.open("r") as file:
            specs = yaml.safe_load(file)

        self.instruction_name = Util.check_instruction_type(specs, DataSimulationTool.__name__, self.expected_instruction)
        self.export_format = Util.check_export_format(specs, DataSimulationTool.__name__, self.instruction_name)

        ParameterValidator.assert_keys_present(specs["definitions"], ["datasets"], DataSimulationTool.__name__, "definitions/datasets")
        ParameterValidator.assert_type_and_value(specs['definitions']['datasets'], dict, DataSimulationTool.__name__, "definitions/datasets")

        dataset_names = list(specs['definitions']['datasets'].keys())
        assert len(dataset_names) == 1, f"{DataSimulationTool.__name__}: one dataset has to be defined under definitions/datasets, got " \
                                        f"{dataset_names} instead."

        self.dataset_name = dataset_names[0]

        Util.check_paths(specs, DataSimulationTool.__name__)
        Util.update_result_paths(specs, self.result_path, self.yaml_path)
