from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.ParameterParser import ParameterParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.preprocessing.Preprocessor import Preprocessor
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
        params = DefaultParamsLoader.load(ImportParser.keyword, dataset_specs["format"])
        if "params" in dataset_specs.keys():
            params = {**params, **dataset_specs["params"]}
        workflow_specification[ImportParser.keyword][key]["params"] = params  # update params with defaults
        dataset = loader().load(dataset_specs["path"], params)

        dataset = ImportParser._preprocess(dataset, dataset_specs)

        symbol_table.add(key, SymbolType.DATASET, dataset)
        return symbol_table

    @staticmethod
    def _preprocess(dataset: RepertoireDataset, workflow_specification: dict) -> RepertoireDataset:
        if "preprocessing" in workflow_specification.keys():
            for key in workflow_specification["preprocessing"].keys():
                preproc_class, params, dataset_specs = ImportParser._parse_preprocessor(workflow_specification["preprocessing"][key]["type"], workflow_specification["preprocessing"][key].get("params", {}))
                workflow_specification["preprocessing"][key]["params"] = dataset_specs  # update params with defaults
                dataset = preproc_class.process(dataset, params)
        return dataset

    @staticmethod
    def _parse_preprocessor(preprocessor_class_name: str, preprocessor_params: dict):
        preprocessor_class = ImportParser._get_preprocessor_class(preprocessor_class_name)
        params, params_specs = ParameterParser.parse(preprocessor_params, preprocessor_class_name, "import_parsers/")
        return preprocessor_class, params, params_specs

    @staticmethod
    def _get_preprocessor_class(name) -> Preprocessor:
        return ReflectionHandler.get_class_by_name(name, "preprocessing/")
