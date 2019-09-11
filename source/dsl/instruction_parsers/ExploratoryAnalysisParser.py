from source.dsl.SymbolTable import SymbolTable
from source.environment.LabelConfiguration import LabelConfiguration
from source.workflows.processes.exploratory_analysis.ExploratoryAnalysisProcess import ExploratoryAnalysisProcess
from source.workflows.processes.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class ExploratoryAnalysisParser:

    """

    The specification consists of a list of analyses that need to be performed;

    Each analysis is defined by a dataset identifier, a report identifier and optionally encoding and labels
    and are loaded into ExploratoryAnalysisUnit objects;

    DSL example for ExploratoryAnalysisProcess assuming that d1, r1, r2, e1 are defined previously in definitions section:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: ExploratoryAnalysis
            analyses:
                -   dataset: d1
                    report: r1
                -   dataset: d1
                    encoding: e1
                    report: r2
                    labels:
                        - CD
                        - CMV

    """

    def parse(self, instruction: dict, symbol_table: SymbolTable) -> ExploratoryAnalysisProcess:
        exp_analysis_units = []
        for analysis in instruction["analyses"]:

            params = self._prepare_params(analysis, symbol_table)
            exp_analysis_units.append(ExploratoryAnalysisUnit(**params))

        process = ExploratoryAnalysisProcess(exploratory_analysis_units=exp_analysis_units)
        return process

    def _prepare_params(self, analysis: dict, symbol_table: SymbolTable) -> dict:
        params = {"dataset": symbol_table.get(analysis["dataset"]), "report": symbol_table.get(analysis["report"])}

        optional_params = self._prepare_optional_params(analysis, symbol_table)
        params = {**params, **optional_params}

        return params

    def _prepare_optional_params(self, analysis: dict, symbol_table: SymbolTable) -> dict:

        params = {}
        dataset = symbol_table.get(analysis["dataset"])

        if all(key in analysis for key in ["encoding", "labels"]):
            params["encoder"] = symbol_table.get(analysis["encoding"]) \
                .create_encoder(dataset, symbol_table.get_config(analysis["encoding"])["encoder_params"])
            params["label_config"] = LabelConfiguration()
            for label in analysis["labels"]:
                params["label_config"].add_label(label, dataset.params[label])
        elif any(key in analysis for key in ["encoding", "labels"]):
            raise KeyError("ExploratoryAnalysisParser: keys for analyses are not properly defined. "
                           "If encoding is defined, labels have to be defined as well and vice versa.")

        if "preprocessing_sequence" in analysis:
            params["preprocessing_sequence"] = symbol_table.get(analysis["preprocessing_sequence"])

        return params
