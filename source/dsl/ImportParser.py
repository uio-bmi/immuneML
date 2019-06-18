from source.data_model.dataset.Dataset import Dataset
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


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
    def _preprocess(dataset: Dataset, dataset_specs: dict) -> Dataset:

        if "preprocessing" in dataset_specs.keys():
            for key in dataset_specs["preprocessing"].keys():

                preproc_class = ReflectionHandler.get_class_by_name(dataset_specs["preprocessing"][key]["type"])

                params = DefaultParamsLoader.load("preprocessing/", dataset_specs["preprocessing"][key]["type"])
                if "params" in dataset_specs["preprocessing"][key]:
                    params = {**params, **dataset_specs["preprocessing"][key]["params"]}
                dataset_specs["preprocessing"][key]["params"] = params  # update params with defaults

                dataset = preproc_class.process(dataset, params)

        return dataset
