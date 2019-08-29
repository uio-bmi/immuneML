from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


class ImportParser:

    keyword = "datasets"

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        assert ImportParser.keyword in workflow_specification, "ImmuneMLParser: datasets are not defined."

        for key in workflow_specification[ImportParser.keyword].keys():
            symbol_table = ImportParser._parse_dataset(workflow_specification, symbol_table, key)

        return symbol_table, workflow_specification[ImportParser.keyword]

    @staticmethod
    def _parse_dataset(workflow_specification: dict, symbol_table: SymbolTable, key: str) -> SymbolTable:

        dataset_specs = workflow_specification[ImportParser.keyword][key]
        loader = ReflectionHandler.get_class_by_name("{}Loader".format(dataset_specs["format"]))

        params = ImportParser._prepare_params(dataset_specs)

        dataset = loader().load(dataset_specs["path"], params)
        symbol_table.add(key, SymbolType.DATASET, dataset)

        return symbol_table

    @staticmethod
    def _prepare_params(dataset_specs: dict) -> dict:
        params = DefaultParamsLoader.load(ImportParser.keyword, dataset_specs["format"])
        if "params" in dataset_specs.keys():
            params = {**params, **dataset_specs["params"]}
        dataset_specs["params"] = params
        return params
