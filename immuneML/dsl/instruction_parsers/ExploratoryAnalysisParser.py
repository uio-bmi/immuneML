import copy
from pathlib import Path

from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class ExploratoryAnalysisParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> ExploratoryAnalysisInstruction:
        exp_analysis_units = {}

        ParameterValidator.assert_keys(instruction, ["analyses", "type", "number_of_processes"],
                                       "ExploratoryAnalysisParser", "ExploratoryAnalysis")
        ParameterValidator.assert_type_and_value(instruction["number_of_processes"], int,
                                                 ExploratoryAnalysisParser.__name__, "number_of_processes")

        for analysis_key, analysis in instruction["analyses"].items():

            params = self._prepare_params(analysis, symbol_table, f"{key}/{analysis_key}")
            params["number_of_processes"] = instruction["number_of_processes"]
            exp_analysis_units[analysis_key] = ExploratoryAnalysisUnit(**params)

        process = ExploratoryAnalysisInstruction(exploratory_analysis_units=exp_analysis_units, name=key)
        return process

    def _prepare_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        valid_keys = ["dataset", "report", "reports", "preprocessing_sequence", "labels", "encoding",
                      "example_weighting", "dim_reduction"] # both report and reports for backwards compatibility
        ParameterValidator.assert_keys(list(analysis.keys()), valid_keys, "ExploratoryAnalysisParser", "analysis", False)

        params = {"dataset": symbol_table.get(analysis["dataset"])}

        if "reports" in analysis:
            params['reports'] = [copy.deepcopy(symbol_table.get(report)) for report in analysis['reports']]
        elif "report" in analysis:
            params['reports'] = [copy.deepcopy(symbol_table.get(analysis['report']))]

        optional_params = self._prepare_optional_params(analysis, symbol_table, yaml_location)
        params = {**params, **optional_params}

        return params

    def _prepare_optional_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        params = {}
        dataset = symbol_table.get(analysis["dataset"])
        loc = ExploratoryAnalysisParser.__name__

        if "encoding" in analysis:
            params["encoder"] = symbol_table.get(analysis["encoding"]).build_object(dataset, **symbol_table.get_config(analysis["encoding"])["encoder_params"])

        if "labels" in analysis:
            params["label_config"] = LabelHelper.create_label_config(analysis["labels"], dataset, loc, yaml_location)
        else:
            params["label_config"] = LabelConfiguration()

        if "preprocessing_sequence" in analysis:
            params["preprocessing_sequence"] = symbol_table.get(analysis["preprocessing_sequence"])

        if "example_weighting" in analysis:
            params["example_weighting"] = symbol_table.get(analysis["example_weighting"]).build_object(dataset, **symbol_table.get_config(analysis["example_weighting"])["example_weighting_params"])

        if "dim_reduction" in analysis:
            valid_dim_reductions = {el.symbol: el.item for el in symbol_table.get_by_type(SymbolType.ML_METHOD)
                                    if isinstance(el.item, DimRedMethod)}
            ParameterValidator.assert_in_valid_list(analysis["dim_reduction"], list(valid_dim_reductions.keys()),
                                                    ExploratoryAnalysisParser.__name__, "dim_reduction")
            params["dim_reduction"] = copy.deepcopy(valid_dim_reductions[analysis['dim_reduction']])

            assert isinstance(params["encoder"], (KmerFrequencyEncoder, Word2VecEncoder)), \
                (f"{loc}: {yaml_location}: Only KmerFrequency and Word2Vec are valid encoders when doing dimensionality"
                 f" reduction.")

        return params
