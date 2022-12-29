import copy
from pathlib import Path

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.dsl.definition_parsers.ReportParser import ReportParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.presentation.html.MultiDatasetBenchmarkHTMLBuilder import MultiDatasetBenchmarkHTMLBuilder
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class MultiDatasetBenchmarkTool:
    """
    MultiDatasetBenchmarkTool trains the models using nested cross-validation (CV) to determine optimal model on multiple datasets. Internally, it uses
    TrainMLModel instruction for each of the listed datasets and performs nested CV on each, accumulates the results of these runs and then
    generates reports on the cumulative results.

    YAML specification:

    .. highlight:: yaml
    .. code-block:: yaml

        definitions: # everything under definitions can be defined in a standard way
            datasets:
                d1: ...
                d2: ...
                d3: ...
            ml_methods:
                ml1: ...
                ml2: ...
                ml3: ...
            encodings:
                enc1: ...
                enc2: ...
            reports:
                r1: ...
                r2: ...
        instructions: # there can be only one instruction
            benchmark_instruction:
                type: TrainMLModel
                benchmark_reports: [r1, r2] # list of reports that will be executed on the results for all datasets
                datasets: [d1, d2, d3] # the same optimization will be performed separately for each dataset
                settings: # a list of combinations of preprocessing, encoding and ml_method to optimize over
                - encoding: enc1 # mandatory field
                  ml_method: ml1 # mandatory field
                - encoding: enc2
                  ml_method: ml2
                - encoding: enc2
                  ml_method: ml3
                assessment: # outer loop of nested CV
                    split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
                    split_count: 1 # how many train/test datasets to generate
                    training_percentage: 0.7 # what percentage of the original data should be used for the training set
                selection: # inner loop of nested CV
                    split_strategy: k_fold # perform k-fold CV
                    split_count: 5 # how many fold to create: here these two parameters mean: do 5-fold CV
                labels: # list of labels to optimize the classifier for, as given in the metadata for the dataset
                    - celiac
                strategy: GridSearch # how to choose the combinations which to test from settings (GridSearch means test all)
                metrics: # list of metrics to compute for all settings, but these do not influence the choice of optimal model
                    - accuracy
                    - auc
                reports: # reports to execute on the dataset (before CV, splitting, encoding etc.)
                    - rep1
                number_of_processes: 4 # number of parallel processes to create (could speed up the computation)
                optimization_metric: balanced_accuracy # the metric to use for choosing the optimal model and during training

    """

    def __init__(self, specification_path: Path, result_path: Path, **kwargs):
        self.specification_path = specification_path
        self.result_path = result_path
        self.reports = None

    def run(self):
        print("Starting MultiDatasetBenchmarkTool...", flush=True)
        PathBuilder.build(self.result_path)
        specs = self._split_specs_file()
        self._extract_reports()
        instruction_states = {}
        for index, specs_name in enumerate(specs.keys()):
            print(f"Running nested cross-validation on dataset {specs_name} ({index+1}/{len(list(specs.keys()))})..", flush=True)
            app = ImmuneMLApp(specification_path=specs[specs_name], result_path=self.result_path / specs_name)
            instruction_states[specs_name] = app.run()[0]
            print(f"Finished nested cross-validation on dataset {specs_name} ({index+1}/{len(list(specs.keys()))})..", flush=True)

        print("Running reports on the results of nested cross-validation on all datasets...", flush=True)
        report_results = self._run_reports(instruction_states)
        print("Finished reports, now generating HTML output...", flush=True)
        MultiDatasetBenchmarkHTMLBuilder.build(report_results, self.result_path,
                                               {specs_name: self.result_path / specs_name for specs_name in specs.keys()})
        print("MultiDatasetBenchmarkTool finished.", flush=True)

    def _extract_reports(self):
        with self.specification_path.open("r") as file:
            workflow_specification = yaml.safe_load(file)

        report_keys = list(workflow_specification['instructions'].values())[0]['benchmark_reports']

        ParameterValidator.assert_all_in_valid_list(report_keys, list(workflow_specification['definitions']['reports'].keys()),
                                                    MultiDatasetBenchmarkTool.__name__, "benchmark_reports")

        reports = {key: value for key, value in workflow_specification['definitions']['reports'].items() if key in report_keys}
        symbol_table, _ = ReportParser.parse_reports(reports, SymbolTable())
        self.reports = [entry.item for entry in symbol_table.get_by_type(SymbolType.REPORT)]

    def _split_specs_file(self) -> dict:
        with self.specification_path.open("r") as file:
            workflow_specification = yaml.safe_load(file)

        self._check_specs(workflow_specification)

        specs_files = {}

        instruction_name = list(workflow_specification['instructions'].keys())[0]
        instruction = workflow_specification['instructions'][instruction_name]

        for dataset_name in instruction['datasets']:
            new_specs = copy.deepcopy(workflow_specification)
            new_specs['definitions']['datasets'] = {dataset_name: new_specs['definitions']['datasets'][dataset_name]}
            del new_specs['instructions'][instruction_name]['datasets']
            del new_specs['instructions'][instruction_name]['benchmark_reports']
            new_specs['instructions'][instruction_name]['dataset'] = dataset_name
            new_specs_file = self.result_path / f"specs_{dataset_name}.yaml"
            with new_specs_file.open('w') as file:
                yaml.dump(new_specs, file)
            specs_files[dataset_name] = new_specs_file

        return specs_files

    def _check_specs(self, workflow_specification):
        location = 'MultiDatasetBenchmarkTool'
        ParameterValidator.assert_keys(workflow_specification.keys(), ['definitions', 'instructions', 'output'], location, 'YAML specification')

        self._check_dataset_specs(workflow_specification, location)
        self._check_instruction_specs(workflow_specification, location)

    def _check_dataset_specs(self, workflow_specification, location):
        ParameterValidator.assert_type_and_value(workflow_specification['definitions'], dict, location, 'definitions')
        ParameterValidator.assert_keys_present(workflow_specification['definitions'].keys(), ['datasets'], location, 'definitions')
        ParameterValidator.assert_type_and_value(workflow_specification['definitions']['datasets'], dict, location, 'datasets')

        dataset_names = list(workflow_specification['definitions']['datasets'].keys())

        assert len(dataset_names) > 1, \
            f"MultiDatasetBenchmarkTool: there is only one dataset specified ({dataset_names[0]}), while this tool operates on multiple datasets. " \
            f"If only one dataset is needed, consider using the training instruction directly."

    def _check_instruction_specs(self, workflow_specification, location):
        ParameterValidator.assert_type_and_value(workflow_specification['instructions'], dict, location, 'instructions')

        instruction_names = list(workflow_specification['instructions'].keys())
        assert len(instruction_names) == 1, f"MultiDatasetBenchmarkTool: there can be only one instruction specified for this tool. " \
                                            f"Currently the following instructions are specified: {instruction_names}."

        ParameterValidator.assert_keys_present(workflow_specification['instructions'][instruction_names[0]].keys(), ['type', 'datasets'], location,
                                               instruction_names[0])

        instruction_type = workflow_specification['instructions'][instruction_names[0]]['type']
        assert instruction_type == 'TrainMLModel', \
            f"MultiDatasetBenchmarkTool: this tool works only with instruction of type 'TrainMLModel', got {instruction_type} instead."

        datasets_in_instruction = workflow_specification['instructions'][instruction_names[0]]['datasets']
        assert len(datasets_in_instruction) > 1, \
            f'{location}: this tool takes a multiple dataset names as input, but only {len(datasets_in_instruction)} were provided: ' \
            f'{datasets_in_instruction}.'

    def _run_reports(self, instruction_states: dict):
        report_results = {}
        for index, report in enumerate(self.reports):
            print(f"Running report {report.name} ({index+1}/{len(self.reports)})...", flush=True)
            report.instruction_states = list(instruction_states.values())
            report.result_path = PathBuilder.build(self.result_path / 'benchmarking_reports/')
            report_result = report.generate_report()
            report_results[report.name] = report_result

        return report_results
