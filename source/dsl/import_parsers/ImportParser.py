from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.logging.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class ImportParser:

    keyword = "datasets"
    valid_keys = ["format", "path", "params", "result_path"]

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        assert ImportParser.keyword in workflow_specification, "ImmuneMLParser: datasets are not defined."

        for key in workflow_specification[ImportParser.keyword].keys():
            symbol_table = ImportParser._parse_dataset(key, workflow_specification[ImportParser.keyword][key], symbol_table)

        return symbol_table, workflow_specification[ImportParser.keyword]

    @staticmethod
    @log
    def _parse_dataset(key: str, dataset_specs: dict, symbol_table: SymbolTable) -> SymbolTable:
        # TODO: update after loader refactor: https://trello.com/c/CsRHADbT
        location = "ImportParser"

        ParameterValidator.assert_keys(list(dataset_specs.keys()), ImportParser.valid_keys, location, f"datasets-{key}", False)

        valid_formats = [name[:-6] for name in ReflectionHandler.discover_classes_by_partial_name("Loader", "IO/dataset_import/")]
        ParameterValidator.assert_in_valid_list(dataset_specs["format"], valid_formats, location, "format")

        loader = ReflectionHandler.get_class_by_name("{}Loader".format(dataset_specs["format"]))
        params = ImportParser._prepare_params(dataset_specs)

        try:
            dataset = loader().load(dataset_specs["path"], params)
            symbol_table.add(key, SymbolType.DATASET, dataset)
        except KeyError as key_error:
            print(f"An error occurred during parsing of dataset {key}. "
                  f"Parameter {key_error.args[0]} was not defined in definitions:datasets:{key}:params.")
            raise

        return symbol_table

    @staticmethod
    def _prepare_params(dataset_specs: dict) -> dict:
        params = DefaultParamsLoader.load(ImportParser.keyword, dataset_specs["format"])
        if "params" in dataset_specs.keys():
            params = {**params, **dataset_specs["params"]}
        dataset_specs["params"] = params
        return params
