from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState


class ExploratoryAnalysisHTMLBuilder:
    """
    A class that will make a HTML file(s) out of ExploratoryAnalysisState object to show what analysis took place in
    the ExploratoryAnalysisInstruction.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: ExploratoryAnalysisState) -> Path:
        """
        Function that builds the HTML files based on the ExploratoryAnalysis state.
        Arguments:
            state: ExploratoryAnalysisState object with details and results of the instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = ExploratoryAnalysisHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"ExploratoryAnalysis_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "ExploratoryAnalysis.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: ExploratoryAnalysisState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ExploratoryAnalysisHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            'immuneML_version': MLUtil.get_immuneML_version(),
            "analyses": [
                {
                    **Util.make_dataset_html_map(analysis.dataset),
                    **{
                        "name": name,
                        "preprocessing_sequence": [
                            {
                                "preprocessing_name": preprocessing.__class__.__name__,
                                "preprocessing_params": ", ".join(
                                    [f"{key}: {value}" for key, value in vars(preprocessing).items()])
                            } for preprocessing in analysis.preprocessing_sequence
                        ] if analysis.preprocessing_sequence is not None else [],
                        "show_preprocessing": analysis.preprocessing_sequence is not None and len(
                            analysis.preprocessing_sequence) > 0,
                        "show_labels": analysis.label_config is not None and len(
                            analysis.label_config.get_labels_by_name()) > 0,
                        "analysis_labels": [{"name": label.name, "values": str(label.values)[1:-1]}
                                   for label in
                                   analysis.label_config.get_label_objects()] if analysis.label_config else None,
                        "encoding_key": analysis.encoder.name if analysis.encoder is not None else None,
                        "encoding_name": StringHelper.camel_case_to_word_string(
                            type(analysis.encoder).__name__) if analysis.encoder is not None
                        else None,
                        "encoding_params": [{"param_name": key, "param_value": str(value)} for key, value in
                                            vars(analysis.encoder).items()] if analysis.encoder is not None else None,
                        "show_encoding": analysis.encoder is not None,
                        "report": Util.to_dict_recursive(Util.update_report_paths(analysis.report_result, base_path),
                                                         base_path) if analysis.report_result is not None else None
                    }
                } for name, analysis in state.exploratory_analysis_units.items()]
        }

        for analysis in html_map["analyses"]:
            if analysis["report"] is not None:
                analysis["show_tables"] = len(analysis["report"]["output_tables"]) > 0 if "output_tables" in analysis["report"] else False
                analysis["show_text"] = len(analysis["report"]["output_text"]) > 0 if "output_text" in analysis["report"] else False
                analysis["show_info"] = analysis["report"]["info"] is not None and len(analysis["report"]["info"]) > 0 if "info" in analysis["report"] else False
            else:
                analysis["show_tables"] = False
                analysis["show_text"] = False
                analysis["show_info"] = False

        return html_map
