import copy

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.ParameterValidator import ParameterValidator
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class ExploratoryAnalysisParser:

    """

    The specification consists of a list of analyses that need to be performed;

    Each analysis is defined by a dataset identifier, a report identifier and optionally encoding and labels
    and are loaded into ExploratoryAnalysisUnit objects;

    DSL example for ExploratoryAnalysisInstruction assuming that d1, r1, r2, e1 are defined previously in definitions section:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: ExploratoryAnalysis
            analyses:
                my_first_analysis:
                    dataset: d1
                    report: r1
                my_second_analysis:
                    dataset: d1
                    encoding: e1
                    report: r2
                    labels:
                        - CD
                        - CMV

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: str = None) -> ExploratoryAnalysisInstruction:
        exp_analysis_units = {}

        ParameterValidator.assert_keys(instruction, ["analyses", "type"], "ExploratoryAnalysisParser", "ExploratoryAnalysis")
        for analysis_key, analysis in instruction["analyses"].items():

            params = self._prepare_params(analysis, symbol_table)
            exp_analysis_units[analysis_key] = ExploratoryAnalysisUnit(**params)

        process = ExploratoryAnalysisInstruction(exploratory_analysis_units=exp_analysis_units, name=key)
        return process

    def _prepare_params(self, analysis: dict, symbol_table: SymbolTable) -> dict:

        valid_keys = ["dataset", "report", "preprocessing_sequence", "labels", "encoding", "batch_size"]
        ParameterValidator.assert_keys(list(analysis.keys()), valid_keys, "ExploratoryAnalysisParser", "analysis", False)

        params = {"dataset": symbol_table.get(analysis["dataset"]), "report": copy.deepcopy(symbol_table.get(analysis["report"]))}

        optional_params = self._prepare_optional_params(analysis, symbol_table)
        params = {**params, **optional_params}

        return params

    def _get_label_values(self, label, dataset):
        if isinstance(dataset, RepertoireDataset):
            values = list(set(dataset.get_metadata([label])[label]))
        elif label in dataset.params:
            values = dataset.params[label]
        else:
            values = []
        return values

    def _prepare_optional_params(self, analysis: dict, symbol_table: SymbolTable) -> dict:

        params = {}
        dataset = symbol_table.get(analysis["dataset"])

        if all(key in analysis for key in ["encoding", "labels"]):
            params["encoder"] = symbol_table.get(analysis["encoding"]) \
                .build_object(dataset, **symbol_table.get_config(analysis["encoding"])["encoder_params"])
            params["label_config"] = LabelConfiguration()
            for label in analysis["labels"]:
                label_values = self._get_label_values(label, dataset)
                params["label_config"].add_label(label, label_values)
        elif any(key in analysis for key in ["encoding", "labels"]):
            raise KeyError("ExploratoryAnalysisParser: keys for analyses are not properly defined. "
                           "If encoding is defined, labels have to be defined as well and vice versa.")

        if "preprocessing_sequence" in analysis:
            params["preprocessing_sequence"] = symbol_table.get(analysis["preprocessing_sequence"])

        if "batch_size" in analysis:
            params["batch_size"] = analysis["batch_size"]

        return params
