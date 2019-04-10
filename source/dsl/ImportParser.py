from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


class ImportParser:

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        assert "dataset_import" in workflow_specification, "Parser: dataset import is not defined."
        for key in workflow_specification["dataset_import"].keys():
            symbol_table = ImportParser._parse_dataset(workflow_specification, symbol_table, key)
        return symbol_table, {}

    @staticmethod
    def _parse_dataset(workflow_specification: dict, symbol_table: SymbolTable, key: str) -> SymbolTable:
        dataset_specs = workflow_specification["dataset_import"][key]
        loader = ReflectionHandler.get_class_by_name("{}Loader".format(dataset_specs["format"]))
        params = {}
        if "params" in dataset_specs.keys():
            params = dataset_specs["params"]
        dataset = loader.load(dataset_specs["path"], params)

        symbol_table.add(key, SymbolType.DATASET, {"dataset": dataset})
        return symbol_table

