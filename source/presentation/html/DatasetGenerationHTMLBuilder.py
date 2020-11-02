import os

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder
from source.util.StringHelper import StringHelper
from source.workflows.instructions.dataset_generation.DatasetGenerationState import DatasetGenerationState


class DatasetGenerationHTMLBuilder:

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(state: DatasetGenerationState) -> str:
        """
        Function that builds the HTML files based on the Simulation state.
        Arguments:
            state: SimulationState object including all details of the Simulation instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path + "../HTML_output/")
        html_map = DatasetGenerationHTMLBuilder.make_html_map(state, base_path)
        result_file = f"{base_path}DatasetGeneration_{state.name}.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}DatasetGeneration.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: DatasetGenerationState, base_path: str) -> dict:
        html_map = {
            "css_style": Util.get_css_content(DatasetGenerationHTMLBuilder.CSS_PATH),
            "name": state.name,
            "full_specs": Util.get_full_specs_path(base_path),
            "datasets": [
                {
                    "dataset_name": dataset.name,
                    "dataset_type": StringHelper.camel_case_to_word_string(type(dataset).__name__),
                    "dataset_size": f"{dataset.get_example_count()} {type(dataset).__name__.replace('Dataset', 's').lower()}",
                    "formats": [
                        {
                            "format_name": format_name,
                            "dataset_download_link": os.path.relpath(path=Util.make_downloadable_zip(state.result_path,
                                                                                                     state.paths[dataset.name][format_name]),
                                                                     start=base_path)
                        } for format_name in state.formats
                    ]
                } for dataset in state.datasets
            ]
        }

        return html_map
