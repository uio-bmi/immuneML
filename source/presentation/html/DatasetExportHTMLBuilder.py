import os
from pathlib import Path

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.util.Util import Util as MLUtil
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder
from source.util.StringHelper import StringHelper
from source.workflows.instructions.dataset_generation.DatasetExportState import DatasetExportState


class DatasetExportHTMLBuilder:

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: DatasetExportState) -> Path:
        """
        Function that builds the HTML files based on the Simulation state.
        Arguments:
            state: SimulationState object including all details of the Simulation instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = DatasetExportHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"DatasetExport_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "DatasetExport.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: DatasetExportState, base_path: str) -> dict:
        html_map = {
            "css_style": Util.get_css_content(DatasetExportHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "datasets": [
                {
                    "dataset_name": dataset.name,
                    "dataset_type": StringHelper.camel_case_to_word_string(type(dataset).__name__),
                    "dataset_size": f"{dataset.get_example_count()} {type(dataset).__name__.replace('Dataset', 's').lower()}",
                    "formats": [
                        {
                            "format_name": format_name,
                            "dataset_download_link": Path(os.path.relpath(path=Util.make_downloadable_zip(state.result_path,
                                                                                                     state.paths[dataset.name][format_name]),
                                                                     start=base_path))
                        } for format_name in state.formats
                    ]
                } for dataset in state.datasets
            ]
        }

        return html_map
