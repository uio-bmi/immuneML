import shutil

import yaml

from source.api.galaxy.Util import Util
from source.app.ImmuneMLApp import ImmuneMLApp
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder


class DatasetGenerationTool:
    """
    DatasetGenerationTool is an alternative to running ImmuneMLApp directly. It accepts a path to YAML specification and a path to the
    output directory and generates the dataset according to the given specification. The created dataset will be located under
    supplied output directory, under results folder. The main dataset file will have the name of the dataset given in the
    specification and has an extension .iml_dataset.

    This tool is meant to be used as an endpoint for Galaxy tool that will create a Galaxy collection out of a dataset in immuneML format.

    Specification for this tool is the same as for the `DatasetGenerationInstruction`, except it can create only one dataset with one format at
    the time.

    """

    def __init__(self, specification_path, result_path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, "Dataset generation tool")

        self.yaml_path = specification_path
        self.result_path = result_path if result_path[-1] == '/' else f"{result_path}/"
        self.files_path = "./"

    def run(self):
        PathBuilder.build(self.result_path)
        self.update_specs()
        state = ImmuneMLApp(self.yaml_path, self.result_path).run()[0]
        shutil.copytree(list(list(state.paths.values())[0].values())[0], self.result_path + "result/")
        print("Exported dataset.")

    def update_specs(self):
        with open(self.yaml_path, 'r') as file:
            specs = yaml.safe_load(file)

        ParameterValidator.assert_keys(specs.keys(), ['definitions', 'instructions'], DatasetGenerationTool.__name__, 'YAML specification')

        self._check_dataset(specs)
        self._check_instruction(specs)

        Util.check_paths(specs, DatasetGenerationTool.__name__)
        Util.update_result_paths(specs, self.result_path, self.yaml_path)

    def _check_dataset(self, specs):
        ParameterValidator.assert_keys_present(specs["definitions"].keys(), ['datasets'], DatasetGenerationTool.__name__, 'definitions')
        assert len(specs['definitions']['datasets'].keys()) == 1, \
            f"{DatasetGenerationTool.__name__}: only one dataset can be defined with this Galaxy tool, got these " \
            f"instead: {list(specs['definitions']['datasets'].keys())}."

        assert len(specs['instructions'].keys()) == 1, \
            f"{DatasetGenerationTool.__name__}: only one instruction of type DatasetGeneration can be defined with this Galaxy tool, got these " \
            f"instructions instead: {list(specs['instructions'].keys())}."

    def _check_instruction(self, specs):
        instruction_name = list(specs['instructions'].keys())[0]

        ParameterValidator.assert_keys_present(specs['instructions'][instruction_name], ['type'], DatasetGenerationTool.__name__, instruction_name)

        assert specs['instructions'][instruction_name]['type'] == "DatasetGeneration", \
            f"{DatasetGenerationTool.__name__}: the instruction has to be of type DatasetGeneration, " \
            f"got {specs['instructions'][instruction_name]['type']} instead."

        for key in ['datasets', 'formats']:
            ParameterValidator.assert_keys_present(specs['instructions'][instruction_name].keys(), [key], DatasetGenerationTool.__name__,
                                                   instruction_name)
            ParameterValidator.assert_type_and_value(specs["instructions"][instruction_name][key], list, DatasetGenerationTool.__name__,
                                                     f"{instruction_name}/{key}")

            assert len(specs['instructions'][instruction_name][key]) == 1, \
                f"{DatasetGenerationTool.__name__}: this tool accepts only one item under {key}, got {specs['instructions'][instruction_name][key]} " \
                f"instead."
