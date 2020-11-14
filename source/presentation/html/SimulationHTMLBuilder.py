from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.util.Util import Util as MLUtil
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.simulation.SimulationState import SimulationState
from source.util.PathBuilder import PathBuilder
from source.util.StringHelper import StringHelper


class SimulationHTMLBuilder:
    """
    A class that will make a HTML file(s) out of SimulationState object to show what analysis took place in
    the SimulationInstruction.
    """

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(state: SimulationState) -> str:
        """
        Function that builds the HTML files based on the Simulation state.
        Arguments:
            state: SimulationState object including all details of the Simulation instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path + "../HTML_output/")
        html_map = SimulationHTMLBuilder.make_html_map(state, base_path)
        result_file = f"{base_path}Simulation_{state.name}.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}Simulation.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: SimulationState, base_path: str) -> dict:

        html_map = {
            "css_style": Util.get_css_content(SimulationHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "dataset_name": state.dataset.name if state.dataset.name is not None else state.dataset.identifier,
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.dataset).__name__),
            "example_count": state.dataset.get_example_count(),
            "implantings": [Util.to_dict_recursive(implanting, base_path) for implanting in state.simulation.implantings]
        }

        return html_map
