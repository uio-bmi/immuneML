import os
from pathlib import Path

from immuneML.data_model.bnp_util import read_yaml
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
        html_map = {**{
            "css_style": Util.get_css_content(LIgOSimulationHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            "show_dataset_labels": state.resulting_dataset.labels is not None,
            "formats": [
                {
                    "format_name": format_name,
                    "dataset_download_link": os.path.relpath(
                        path=Util.make_downloadable_zip(state.result_path,
                                                        state.paths[state.resulting_dataset.name][format_name]),
                        start=base_path)
                } for format_name in state.formats
            ],
            "simulation_details": LIgOSimulationHTMLBuilder.prepare_simulation_details(base_path)
        }, **Util.make_dataset_html_map(state.resulting_dataset)}

        if html_map['show_dataset_labels']:
            html_map['dataset_labels'] = [
                {"dataset_label_name": k, 'dataset_label_classes': str(v)}
                for k, v in state.resulting_dataset.labels.items()
            ]

        return html_map

    @staticmethod
    def prepare_simulation_details(base_path):
        specs = read_yaml(Path(list(base_path.glob("../**/full*.yaml"))[0]))
        sim_details = {
            'motifs': specs['definitions']['motifs'],
            'signals': specs['definitions']['signals'],
            'simulations': specs['definitions']['simulations']
        }
        return build_html(sim_details)

def build_html(d):
    html_output = '<ul>'
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively build the HTML for it
            html_output += f'<li>{key}: {build_html(value)}</li>'
        else:
            # If the value is not a dictionary, display it as a simple list item
            html_output += f'<li>{key}: {value}</li>'
    html_output += '</ul>'
    return html_output
