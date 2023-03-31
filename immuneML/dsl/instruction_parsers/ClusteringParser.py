import copy
from pathlib import Path

from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringInstruction
from immuneML.workflows.instructions.clustering.ClusteringUnit import ClusteringUnit


class ClusteringParser:
    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> ClusteringInstruction:
        clustering_units = {}

        ParameterValidator.assert_keys(instruction, ["analyses", "type", "number_of_processes"], "ClusteringParser", "Cluster")
        ParameterValidator.assert_type_and_value(instruction["number_of_processes"], int, ClusteringParser.__name__, "number_of_processes")

        for analysis_key, analysis in instruction["analyses"].items():
            params = self._prepare_params(analysis, symbol_table, f"{key}/{analysis_key}")
            params["number_of_processes"] = instruction["number_of_processes"]
            clustering_units[analysis_key] = ClusteringUnit(**params)

        process = ClusteringInstruction(clustering_units=clustering_units, name=key)
        return process

    def _prepare_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        valid_keys = ["dataset", "report", "clustering_method", "encoding", "labels", "dimensionality_reduction", "dim_red_before_clustering", "true_labels_path", "eval_metrics"]
        ParameterValidator.assert_keys(list(analysis.keys()), valid_keys, "ClusteringParser", yaml_location[yaml_location.rfind("/")+1:], False)
        must_have_keys = ["dataset", "report", "clustering_method", "encoding"]
        ParameterValidator.assert_keys_present(list(analysis.keys()), must_have_keys, "ClusteringParser", yaml_location[yaml_location.rfind("/")+1:])

        dataset = symbol_table.get(analysis["dataset"])
        params = {"dataset": dataset,
                  "report": copy.deepcopy(symbol_table.get(analysis["report"])),
                  "clustering_method": symbol_table.get(analysis["clustering_method"]),
                  "encoder": symbol_table.get(analysis["encoding"]).build_object(dataset, **symbol_table.get_config(analysis["encoding"])["encoder_params"])}

        params["clustering_method"].check_encoder_compatibility(params["encoder"])

        optional_params = self._prepare_optional_params(analysis, symbol_table, yaml_location)

        if "dimensionality_reduction" in optional_params:
            optional_params["dimensionality_reduction"].check_encoder_compatibility(params["encoder"])

        params = {**params, **optional_params}

        return params

    def _prepare_optional_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:
        params = {}
        dataset = symbol_table.get(analysis["dataset"])

        if "dimensionality_reduction" in analysis:
            params["dimensionality_reduction"] = symbol_table.get(analysis["dimensionality_reduction"])

        if "dim_red_before_clustering" in analysis:
            params["dim_red_before_clustering"] = analysis["dim_red_before_clustering"]

        if "labels" in analysis:
            params["label_config"] = LabelHelper.create_label_config(analysis["labels"], dataset, ClusteringParser.__name__, yaml_location)
        else:
            params["label_config"] = LabelConfiguration()

        if "true_labels_path" in analysis:
            params["true_labels_path"] = Path(analysis["true_labels_path"])

        if "eval_metrics" in analysis:
            params["eval_metrics"] = analysis["eval_metrics"]
        else:
            params["eval_metrics"] = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]

        return params
