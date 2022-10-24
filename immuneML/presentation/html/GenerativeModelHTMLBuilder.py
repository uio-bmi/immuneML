from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper
from immuneML.workflows.instructions.generative_model.GenerativeModelState import GenerativeModelState


class GenerativeModelHTMLBuilder:
    """
    A class that will make a HTML file(s) out of ExploratoryAnalysisState object to show what generator took place in
    the ExploratoryAnalysisInstruction.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: GenerativeModelState) -> Path:
        """
        Function that builds the HTML files based on the ExploratoryAnalysis state.
        Arguments:
            state: ExploratoryAnalysisState object with details and results of the instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = GenerativeModelHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"GenerativeModel_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "GenerativeModel.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: GenerativeModelState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(GenerativeModelHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            'immuneML_version': MLUtil.get_immuneML_version(),
            "analyses": [{
                "name": name,
                "alphabet": [{"letter": c} for c in generator.alphabet],
                "positions": [{"pos": i} for i in range(1, len(generator.PWM[0]) + 1)],
                "generated_sequences": [{"seq_nr": ind, "seq": seq} for ind, seq in enumerate(generator.generated_sequences)],
                "PWM": [{"row": [{"weight": weight} for weight in row], "letter": letter} for row, letter in zip(generator.PWM, generator.alphabet)] if generator.PWM is not None else None,
                "dataset_name": generator.dataset.name if generator.dataset.name is not None else generator.dataset.identifier,
                "dataset_type": StringHelper.camel_case_to_word_string(type(generator.dataset).__name__),
                "example_count": generator.dataset.get_example_count(),
                "dataset_size": f"{generator.dataset.get_example_count()} {type(generator.dataset).__name__.replace('Dataset', 's').lower()}",
                "encoding_key": generator.encoder.name if generator.encoder is not None else None,
                "encoding_name": StringHelper.camel_case_to_word_string(type(generator.encoder).__name__) if generator.encoder is not None
                else None,
                "encoding_params": [{"param_name": key, "param_value": str(value)} for key, value in vars(generator.encoder).items()] if generator.encoder is not None else None,
                "show_encoding": generator.encoder is not None,
                "report": Util.to_dict_recursive(Util.update_report_paths(generator.report_result, base_path), base_path)
            } for name, generator in state.generative_model_units.items()]
        }

        for generator in html_map["analyses"]:
            generator["show_tables"] = len(generator["report"]["output_tables"]) > 0 if "output_tables" in generator["report"] else False
            generator["show_text"] = len(generator["report"]["output_text"]) > 0 if "output_text" in generator["report"] else False
            generator["show_info"] = generator["report"]["info"] is not None and len(generator["report"]["info"]) > 0 if "info" in generator["report"] else False

        return html_map
