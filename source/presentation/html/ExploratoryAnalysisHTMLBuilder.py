from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
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
            state:
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        html_map = ExploratoryAnalysisHTMLBuilder.make_html_map(state)
        result_file = f"{state.result_path}ExploratoryAnalysis.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}ExploratoryAnalysis.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: ExploratoryAnalysisState) -> dict:
        html_map = {
            "css_style": Util.get_css_content(ExploratoryAnalysisHTMLBuilder.CSS_PATH),
            "analyses": [{
                "name": name,
                "dataset_name": analysis.dataset.name if analysis.dataset.name is not None else analysis.dataset.identifier,
                "dataset_type": " ".join(StringHelper.camel_case_to_words(type(analysis.dataset).__name__)),
                "example_count": analysis.dataset.get_example_count(),
                "show_labels": analysis.label_config is not None,
                "labels": [{"name": label.name, "values": str(label.values)[1:-1]}
                           for label in analysis.label_config.get_label_objects()] if analysis.label_config else None,
                "encoding_name": " ".join(StringHelper.camel_case_to_words(type(analysis.encoder).__name__)) if analysis.encoder is not None
                                else None,
                "encoding_params": vars(analysis.encoder) if analysis.encoder is not None else None,
                "show_encoding": analysis.encoder is not None,
                "show_encoding_and_labels": analysis.encoder is not None and analysis.label_config is not None,
                "report": Util.to_dict_recursive(analysis.report_result)
            } for name, analysis in state.exploratory_analysis_units.items()]
        }

        return html_map
