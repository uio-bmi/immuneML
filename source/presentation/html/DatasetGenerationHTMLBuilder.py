import os

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.StringHelper import StringHelper
from source.workflows.instructions.dataset_generation.DatasetGenerationState import DatasetGenerationState


class DatasetGenerationHTMLBuilder:

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(state: DatasetGenerationState, is_index: bool = True) -> str:
        """
        Function that builds the HTML files based on the Simulation state.
        Arguments:
            state: SimulationState object including all details of the Simulation instruction
            is_index: bool indicating if the resulting html will be used as index page so that paths should be adjusted
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = os.path.abspath(state.result_path) + "/" if not is_index else os.path.abspath(state.result_path + "/../") + "/"
        html_map = DatasetGenerationHTMLBuilder.make_html_map(state, base_path)
        result_file = f"{state.result_path}DatasetGeneration.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}DatasetGeneration.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: DatasetGenerationState, base_path: str) -> dict:
        html_map = {
            "css_style": Util.get_css_content(DatasetGenerationHTMLBuilder.CSS_PATH),
            "name": state.name,
            "datasets": [
                {
                    "dataset_name": dataset.name,
                    "dataset_type": StringHelper.camel_case_to_word_string(type(dataset).__name__),
                    "formats": [
                        {
                            "format_name": format_name,
                            "dataset_download_link": Util.make_downloadable_zip(base_path, state.paths[dataset.name][format_name])
                        } for format_name in state.formats
                    ]
                } for dataset in state.datasets
            ]
        }

        return html_map
