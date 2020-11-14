from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.util.Util import Util as MLUtil
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder
from source.util.StringHelper import StringHelper
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState


class ExploratoryAnalysisHTMLBuilder:
    """
    A class that will make a HTML file(s) out of ExploratoryAnalysisState object to show what analysis took place in
    the ExploratoryAnalysisInstruction.
    """

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(state: ExploratoryAnalysisState) -> str:
        """
        Function that builds the HTML files based on the ExploratoryAnalysis state.
        Arguments:
            state: ExploratoryAnalysisState object with details and results of the instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path + "../HTML_output")
        html_map = ExploratoryAnalysisHTMLBuilder.make_html_map(state, base_path)
        result_file = f"{base_path}ExploratoryAnalysis_{state.name}.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}ExploratoryAnalysis.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: ExploratoryAnalysisState, base_path: str) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ExploratoryAnalysisHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            'immuneML_version': MLUtil.get_immuneML_version(),
            "analyses": [{
                "name": name,
                "dataset_name": analysis.dataset.name if analysis.dataset.name is not None else analysis.dataset.identifier,
                "dataset_type": StringHelper.camel_case_to_word_string(type(analysis.dataset).__name__),
                "example_count": analysis.dataset.get_example_count(),
                "show_labels": analysis.label_config is not None,
                "labels": [{"name": label.name, "values": str(label.values)[1:-1]}
                           for label in analysis.label_config.get_label_objects()] if analysis.label_config else None,
                "encoding_name": StringHelper.camel_case_to_word_string(type(analysis.encoder).__name__) if analysis.encoder is not None
                else None,
                "encoding_params": vars(analysis.encoder) if analysis.encoder is not None else None,
                "show_encoding": analysis.encoder is not None,
                "show_encoding_and_labels": analysis.encoder is not None and analysis.label_config is not None,
                "report": Util.to_dict_recursive(analysis.report_result, base_path)
            } for name, analysis in state.exploratory_analysis_units.items()]
        }

        return html_map
