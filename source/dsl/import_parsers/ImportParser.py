import glob

from source.data_model.dataset.Dataset import Dataset
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler
from source.preprocessing.Preprocessor import Preprocessor
from source.environment.EnvironmentSettings import EnvironmentSettings


class ImportParser:

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        assert "dataset_import" in workflow_specification, "Parser: dataset import is not defined."
        for key in workflow_specification["dataset_import"].keys():
            symbol_table = ImportParser._parse_dataset(workflow_specification, symbol_table, key)
        return symbol_table, workflow_specification["dataset_import"]

    @staticmethod
    def _parse_dataset(workflow_specification: dict, symbol_table: SymbolTable, key: str) -> SymbolTable:
        dataset_specs = workflow_specification["dataset_import"][key]
        loader = ReflectionHandler.get_class_by_name("{}Loader".format(dataset_specs["format"]))
        params = DefaultParamsLoader.load("dataset_import/", dataset_specs["format"])
        if "params" in dataset_specs.keys():
            params = {**params, **dataset_specs["params"]}
        workflow_specification["dataset_import"][key]["params"] = params  # update params with defaults
        dataset = loader().load(dataset_specs["path"], params)

        dataset = ImportParser._preprocess(dataset, dataset_specs)

        symbol_table.add(key, SymbolType.DATASET, {"dataset": dataset})
        return symbol_table

    @staticmethod
    def _preprocess(dataset: Dataset, workflow_specification: dict) -> Dataset:
        if "preprocessing" in workflow_specification.keys():
            for key in workflow_specification["preprocessing"].keys():
                preproc_class, params, dataset_specs = ImportParser._parse_preprocessor(workflow_specification["preprocessing"][key]["type"], workflow_specification["preprocessing"][key].get("params", {}))
                workflow_specification["preprocessing"][key]["params"] = dataset_specs  # update params with defaults
                dataset = preproc_class.process(dataset, params)
        return dataset

    @staticmethod
    def _parse_preprocessor(preprocessor_class_name: str, preprocessor_params: dict):
        preprocessor_class = ImportParser._get_preprocessor_class(preprocessor_class_name)
        params, params_specs = ImportParser._get_preprocessor_params(preprocessor_params, preprocessor_class_name)
        return preprocessor_class, params, params_specs

    @staticmethod
    def _get_preprocessor_class(name) -> Preprocessor:
        filenames = glob.glob(EnvironmentSettings.root_path + "source/preprocessing/**/{}.py".format(name))
        assert len(filenames) == 1, "EncodingParser: the preprocessor type was not correctly specified."
        return ReflectionHandler.get_class_from_path(filenames[0])

    @staticmethod
    def _get_preprocessor_params(params, preprocessor_type: str):
        parser_class = ReflectionHandler.get_class_by_name("{}Parser".format(preprocessor_type))
        parsed_params, params_specs = parser_class.parse(params)
        return parsed_params, params_specs
