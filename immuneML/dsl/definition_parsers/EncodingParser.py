import inspect

from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class EncodingParser:

    keyword = "encodings"

    @staticmethod
    def parse(encodings: dict, symbol_table: SymbolTable):
        for key in encodings.keys():

            encoder, params = EncodingParser.parse_encoder(key, encodings[key])
            encodings[key] = {encoder.__name__[:-7]: params}
            symbol_table.add(key, SymbolType.ENCODING, encoder, {"encoder_params": params})

        return symbol_table, encodings

    @staticmethod
    def parse_encoder(key: str, specs: dict):
        class_path = "encodings"
        valid_encoders = ReflectionHandler.all_nonabstract_subclass_basic_names(DatasetEncoder, "Encoder", class_path)
        encoder = ObjectParser.get_class(specs, valid_encoders, "Encoder", class_path, "EncodingParser", key)
        params = ObjectParser.get_all_params(specs, class_path, encoder.__name__[:-7], key)

        required_params = [p for p in list(inspect.signature(encoder.__init__).parameters.keys()) if p != "self"]
        ParameterValidator.assert_all_in_valid_list(params.keys(), required_params, "EncoderParser", f"{key}/{encoder.__name__.replace('Encoder', '')}")

        return encoder, params


