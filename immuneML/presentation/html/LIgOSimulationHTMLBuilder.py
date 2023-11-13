import os
from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.simulation.LigoSimState import LigoSimState
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper


class LIgOSimulationHTMLBuilder:
    """
    A class that will make a HTML file(s) out of SimulationState object to show what analysis took place in
    the LIgOSimulationInstruction.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: LigoSimState) -> str:
        """
        Function that builds the HTML files based on the Simulation state.
        Arguments:
            state: SimulationState object including all details of the Simulation instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = LIgOSimulationHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"Simulation_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Simulation.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: LigoSimState, base_path: Path) -> dict:

        html_map = {
            "css_style": Util.get_css_content(LIgOSimulationHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "dataset_name": state.resulting_dataset.name if state.resulting_dataset.name is not None else state.resulting_dataset.identifier,
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.resulting_dataset).__name__),
            "example_count": state.resulting_dataset.get_example_count(),
            "dataset_size": f"{state.resulting_dataset.get_example_count()} {type(state.resulting_dataset).__name__.replace('Dataset', 's').lower()}",
            "labels": [{"label_name": label} for label in state.resulting_dataset.get_label_names()],
            "formats": [
                {
                    "format_name": format_name,
                    "dataset_download_link": os.path.relpath(
                        path=Util.make_downloadable_zip(state.result_path, state.paths[state.resulting_dataset.name][format_name]),
                        start=base_path)
                } for format_name in state.formats
            ],
            "simulation_items": [Util.to_dict_recursive(implanting, base_path) for implanting in state.simulation.sim_items]
        }

        return html_map
