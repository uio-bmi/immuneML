import copy
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringInstruction
from immuneML.workflows.instructions.clustering.ClusteringUnit import ClusteringUnit


class ClusteringParser:
    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> ClusteringInstruction:
        clustering_units = {}

        ParameterValidator.assert_keys(instruction, ["analyses", "type", "number_of_processes"], "ClusterParser", "Cluster")
        ParameterValidator.assert_type_and_value(instruction["number_of_processes"], int, ClusteringParser.__name__, "number_of_processes")

        for analysis_key, analysis in instruction["analyses"].items():

            params = self._prepare_params(analysis, symbol_table, f"{key}/{analysis_key}")
            params["number_of_processes"] = instruction["number_of_processes"]
            clustering_units[analysis_key] = ClusteringUnit(**params)

        process = ClusteringInstruction(clustering_units=clustering_units, name=key)
        return process

    def _prepare_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        valid_keys = ["dataset", "report", "clustering_method", "encoding", "number_of_processes", "dimensionality_reduction"]
        ParameterValidator.assert_keys(list(analysis.keys()), valid_keys, "ClusteringParser", "analysis", False)

        params = {"dataset": symbol_table.get(analysis["dataset"]),
                  "report": copy.deepcopy(symbol_table.get(analysis["report"])),
                  "clustering_method": symbol_table.get(analysis["clustering_method"]),
                  "label_config": LabelConfiguration()}

        optional_params = self._prepare_optional_params(analysis, symbol_table, yaml_location)
        params = {**params, **optional_params}

        return params

    def _prepare_optional_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:
        params = {}
        dataset = symbol_table.get(analysis["dataset"])

        if "encoding" in analysis:
            params["encoder"] = symbol_table.get(analysis["encoding"]).build_object(dataset, **symbol_table.get_config(analysis["encoding"])["encoder_params"])

        if "dimensionality_reduction" in analysis:
            params["dimensionality_reduction"] = symbol_table.get(analysis["dimensionality_reduction"])

        return params
