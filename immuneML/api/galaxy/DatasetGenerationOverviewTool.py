from pathlib import Path

import yaml

from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.api.galaxy.GalaxyTool import GalaxyTool
from immuneML.api.galaxy.Util import Util
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.util.Logger import print_log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import \
    ExploratoryAnalysisInstruction


class DatasetGenerationOverviewTool(GalaxyTool):
    """
    DatasetGenerationTool is an alternative to running ImmuneMLApp directly.
    This tool is meant to be used as an endpoint for Galaxy tool that will create a Galaxy collection out of a dataset in immuneML format.

    This tool accepts a path to a YAML specification which uses a single dataset, and runs the ExploratoryAnalysisInstruction with optional reports.
    The created dataset will be located in the supplied output directory, under the 'results' folder.
    The main dataset file will have the name of the dataset given in the specification and has an extension .yaml.
    """

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        Util.check_parameters(specification_path, result_path, kwargs, "Dataset generation tool")
        super().__init__(specification_path, result_path, **kwargs)

    def _run(self):
        PathBuilder.build(self.result_path)
        self._update_specs()
        state = ImmuneMLApp(self.yaml_path, self.result_path).run()[0]
        dataset = list(state.exploratory_analysis_units.values())[0].dataset

        ImmuneMLExporter.export(dataset, self.result_path / "result/")

        print_log(f"Exported dataset.")

    def _update_specs(self):
        with self.yaml_path.open('r') as file:
            specs = yaml.safe_load(file)

        ParameterValidator.assert_keys_present(specs.keys(), ["definitions", "instructions"], DatasetGenerationOverviewTool.__name__, "YAML specification")
        ParameterValidator.assert_all_in_valid_list(specs.keys(), ["definitions", "instructions", "output"], DatasetGenerationOverviewTool.__name__, "YAML specification")

        self._check_dataset(specs)
        self._check_instruction(specs)

        Util.update_dataset_key(specs, DatasetGenerationOverviewTool.__name__)
        Util.check_paths(specs, DatasetGenerationOverviewTool.__name__)
        Util.update_result_paths(specs, self.result_path, self.yaml_path)

    def _check_dataset(self, specs):
        ParameterValidator.assert_keys_present(specs["definitions"].keys(), ['datasets'], DatasetGenerationOverviewTool.__name__, 'definitions')
        assert len(specs['definitions']['datasets'].keys()) == 1, \
            f"{DatasetGenerationOverviewTool.__name__}: only one dataset can be defined with this Galaxy tool, got these " \
            f"instead: {list(specs['definitions']['datasets'].keys())}."

    def _check_instruction(self, specs):
        assert len(specs['instructions'].keys()) == 1, \
            f"{DatasetGenerationOverviewTool.__name__}: only one instruction of type ExploratoryAnalysis can be defined with this Galaxy tool, got these " \
            f"instructions instead: {list(specs['instructions'].keys())}."

        instruction_name = Util.check_instruction_type(specs, DatasetGenerationOverviewTool.__name__, ExploratoryAnalysisInstruction.__name__[:-11])

        dataset_name = None
        for analysis_key, analysis_specs in specs['instructions'][instruction_name]["analyses"].items():
            ParameterValidator.assert_keys_present(analysis_specs.keys(), ["dataset", "report"],
                                                   DatasetGenerationOverviewTool.__name__, f"{instruction_name}/analyses/{analysis_key}")

            if dataset_name is None:
                dataset_name = analysis_specs["dataset"]
            else:
                assert analysis_specs["dataset"] == dataset_name, f"{DatasetGenerationOverviewTool.__name__}: expected only one dataset name. Found: {dataset_name} and {analysis_specs['dataset']}"
