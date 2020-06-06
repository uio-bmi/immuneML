import os

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.api.galaxy.Util import Util
from source.dsl.ImmuneMLParser import ImmuneMLParser
from source.dsl.import_parsers.ImportParser import ImportParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.util.PathBuilder import PathBuilder


class DatasetGenerationTool:

    def __init__(self, yaml_path, output_dir, **kwargs):
        Util.check_parameters(yaml_path, output_dir, kwargs, "Dataset generation tool")

        inputs = kwargs["inputs"].split(',') if "inputs" in kwargs else None

        self.yaml_path = yaml_path
        self.result_path = output_dir if output_dir[-1] == '/' else f"{output_dir}/"
        self.metadata_file = kwargs["metadata"] if "metadata" in kwargs else None
        self.files_path = f"{os.path.dirname(inputs[0])}/" if "inputs" in kwargs else None

    def run(self):
        PathBuilder.build(self.result_path)
        symbol_table, output_path = ImmuneMLParser.parse_yaml_file(self.yaml_path, self.result_path, parse_func=self.parse_dataset)
        datasets = symbol_table.get_keys_by_type(SymbolType.DATASET)
        assert len(datasets) == 1, f"Dataset generation tool: {len(datasets)} datasets were defined. Please check the input parameters."
        dataset = symbol_table.get(datasets[0])
        PickleExporter.export(dataset=dataset, path=self.result_path)
        print(f"Dataset {dataset.name} generated.")

    def parse_dataset(self, workflow_specification: dict, yaml_path: str, result_path: str):
        keys = list(workflow_specification.keys())
        assert len(keys) == 1, f"Dataset generation tool: {len(keys)} keys were specified in the yaml file, but there should be only one," \
                               f"which will be used as a dataset name. Please see the documentation for generating immuneML datasets."

        assert "params" in workflow_specification[keys[0]], \
            f"Dataset generation tool: the format of the specification is not correct. 'params' key missing under '{keys[0]}'." \
            f"Please see the documentation for generating immuneML datasets."

        workflow_specification[keys[0]]["params"]["result_path"] = self.result_path

        workflow_specification[keys[0]]["params"]["metadata_file"] = self.metadata_file
        workflow_specification[keys[0]]["params"]["path"] = self.files_path
        workflow_specification[keys[0]]["params"]["result_path"] = self.result_path

        symbol_table = SymbolTable()
        return ImportParser.parse({"datasets": workflow_specification}, symbol_table)
