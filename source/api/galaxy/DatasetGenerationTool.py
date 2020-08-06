import os

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.api.galaxy.Util import Util
from source.dsl.ImmuneMLParser import ImmuneMLParser
from source.dsl.import_parsers.ImportParser import ImportParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.util.PathBuilder import PathBuilder


class DatasetGenerationTool:
    """
    DatasetGenerationTool is an alternative to running ImmuneMLApp directly. It accepts a path to YAML specification and a path to the
    output directory and generates the dataset according to the given specification. The created dataset will be located under
    supplied output directory, under results folder. The main dataset file will have the name of the dataset given in the
    specification and has an extension .iml_dataset.

    This tool is meant to be used as an endpoint for Galaxy tool that will create a Galaxy collection out of a dataset in immuneML format.

    Specification supplied for this tool is a simplified version of the immuneML specification. It contains only the definition for
    one dataset in the following format (other options under params are possible as in a standard dataset definition in immuneML, with only
    difference that the files must be in the current working directory, so only file names are given in the specs):

        user_specified_dataset_name:
            format: AdaptiveBiotech # format in which the immune receptor or repertoire files are
            params:
                metadata_file: metadata.csv # metadata filename if the dataset consists of repertoires
                path: ./ # by default those files will be searched for at the current working directory

    """

    def __init__(self, yaml_path, output_dir, **kwargs):
        Util.check_parameters(yaml_path, output_dir, kwargs, "Dataset generation tool")

        inputs = kwargs["inputs"].split(',') if "inputs" in kwargs else None

        self.yaml_path = yaml_path
        self.result_path = output_dir if output_dir[-1] == '/' else f"{output_dir}/"
        self.files_path = f"{os.path.dirname(inputs[0])}/" if "inputs" in kwargs else "./"

    def run(self):
        PathBuilder.build(self.result_path)
        symbol_table, output_path = ImmuneMLParser.parse_yaml_file(self.yaml_path, self.result_path, parse_func=self.parse_dataset)
        datasets = symbol_table.get_keys_by_type(SymbolType.DATASET)
        assert len(datasets) == 1, f"Dataset generation tool: {len(datasets)} datasets were defined. Please check the input parameters."
        dataset = symbol_table.get(datasets[0])
        PickleExporter.export(dataset=dataset, path=self.result_path + "result/")
        print(f"Dataset {dataset.name} generated.")

    def parse_dataset(self, workflow_specification: dict, yaml_path: str, result_path: str):
        keys = list(workflow_specification.keys())
        assert len(keys) == 1, f"Dataset generation tool: {len(keys)} keys ({str(keys)[1:-1]}) were specified in the yaml file, but there should be " \
                               f"only one, which will be used as a dataset name. Please see the documentation for generating immuneML datasets."

        assert "params" in workflow_specification[keys[0]], \
            f"Dataset generation tool: the format of the specification is not correct. 'params' key missing under '{keys[0]}'." \
            f"Please see the documentation for generating immuneML datasets."

        if workflow_specification[keys[0]]["format"] not in ["RandomRepertoireDataset", "RandomReceptorDataset"]:
            workflow_specification[keys[0]]["params"]["path"] = self.files_path

        workflow_specification[keys[0]]["params"]["result_path"] = self.result_path

        symbol_table = SymbolTable()
        return ImportParser.parse({"datasets": workflow_specification}, symbol_table)
