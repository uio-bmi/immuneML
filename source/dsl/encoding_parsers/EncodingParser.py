import glob

from source.dsl.ParameterParser import ParameterParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.encodings.DatasetEncoder import DatasetEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.ReflectionHandler import ReflectionHandler


class EncodingParser:

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable):
        if "encodings" in workflow_specification.keys():
            for key in workflow_specification["encodings"].keys():
                encoder, params, params_specs = EncodingParser.parse_encoder(workflow_specification["encodings"][key]["type"], workflow_specification["encodings"][key]["params"])
                config = {"encoder_params": params}
                if "labels" in workflow_specification["encodings"][key].keys():
                    config["labels"] = workflow_specification["encodings"][key]["labels"]
                workflow_specification["encodings"][key]["params"] = params_specs
                symbol_table.add(key, SymbolType.ENCODING, encoder, config)
        else:
            workflow_specification["encodings"] = {}
        return symbol_table, workflow_specification["encodings"]

    @staticmethod
    def parse_encoder(encoder_class_name: str, encoder_params: dict):
        encoder_class = EncodingParser.get_encoder_class(encoder_class_name)
        params, params_specs = ParameterParser.parse(encoder_params, encoder_class_name, "encoding_parsers/")
        return encoder_class, params, params_specs

    @staticmethod
    def get_encoder_class(name) -> DatasetEncoder:
        filenames = glob.glob(EnvironmentSettings.root_path + "source/encodings/**/{}Encoder.py".format(name))
        assert len(filenames) == 1, "EncodingParser: the encoder type was not correctly specified."
        return ReflectionHandler.get_class_from_path(filenames[0])


