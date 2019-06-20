from source.dsl.import_parsers.PreprocessingParameterParser import PreprocessingParameterParser


class PatientRepertoireCollectorParser(PreprocessingParameterParser):

    @staticmethod
    def parse(params: dict):
        return params, params
