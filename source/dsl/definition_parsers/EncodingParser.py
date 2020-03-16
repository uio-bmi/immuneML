from source.dsl.ObjectParser import ObjectParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.encodings.DatasetEncoder import DatasetEncoder
from source.logging.Logger import log
from source.util.ReflectionHandler import ReflectionHandler


class EncodingParser:

    @staticmethod
    def parse(encodings: dict, symbol_table: SymbolTable):
        for key in encodings.keys():

            encoder, params = EncodingParser.parse_encoder(key, encodings[key])
            symbol_table.add(key, SymbolType.ENCODING, encoder, {"encoder_params": params})

        return symbol_table, encodings

    @staticmethod
    @log
    def parse_encoder(key: str, specs: dict):
        class_path = "encodings"
        classes = ReflectionHandler.get_classes_by_partial_name("Encoder", class_path)
        valid_encoders = [cls.__name__[:-7] for cls in DatasetEncoder.__subclasses__()]
        encoder = ObjectParser.get_class(specs, valid_encoders, "Encoder", class_path, "EncodingParser", key)
        params = ObjectParser.get_all_params(specs, class_path, encoder.__name__[:-7])

        return encoder, params

    @staticmethod
    def parse_encoder_internal(short_class_name: str, encoder_params: dict):
        encoder_class = ReflectionHandler.get_class_by_name(f"{short_class_name}Encoder", "encodings")
        params = ObjectParser.get_all_params({short_class_name: encoder_params}, "encodings", short_class_name)
        return encoder_class, params, params



