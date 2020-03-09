from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.encodings.DatasetEncoder import DatasetEncoder
from source.logging.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class EncodingParser:
    """
    example:

    encodings:
        e1:
            KmerFrequency:
                k: 3
                normalization_type: relative_frequency
    """

    @staticmethod
    def parse(encodings: dict, symbol_table: SymbolTable):
        for key in encodings.keys():

            if isinstance(encodings[key], dict):
                encoder_class_name = list(encodings[key].keys())[0]
                params = encodings[key][encoder_class_name]
            else:
                encoder_class_name = list(encodings[key])[0]
                params = {}

            encoder, params, params_specs = EncodingParser.parse_encoder(key, encoder_class_name, params)
            encodings[key] = {encoder_class_name: params_specs}
            symbol_table.add(key, SymbolType.ENCODING, encoder, {"encoder_params": params})

        return symbol_table, encodings

    @staticmethod
    @log
    def parse_encoder(key: str, encoder_class_name: str, encoder_params: dict):
        return EncodingParser.parse_encoder_internal(encoder_class_name, encoder_params)

    @staticmethod
    def parse_encoder_internal(encoder_class_name: str, encoder_params: dict):
        encoder_class = EncodingParser.get_encoder_class(encoder_class_name)
        params = {**DefaultParamsLoader.load("encodings/", encoder_class_name), **encoder_params}
        return encoder_class, params, params

    @staticmethod
    def get_encoder_class(name) -> DatasetEncoder:
        # TODO: find better way to discover valid options for encoders
        classes = ReflectionHandler.get_classes_by_partial_name("Encoder", "encodings/")
        valid_encoders = [cls.__name__[:-7] for cls in DatasetEncoder.__subclasses__()]
        ParameterValidator.assert_in_valid_list(name, valid_encoders, "EncodingParser", "encoder name")

        return ReflectionHandler.get_class_by_name(f"{name}Encoder")


