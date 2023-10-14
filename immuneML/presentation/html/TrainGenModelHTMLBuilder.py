import itertools
from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.train_gen_model.TrainGenModelInstruction import TrainGenModelState


class TrainGenModelHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: TrainGenModelState) -> Path:
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = TrainGenModelHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"DatasetExport_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "TrainGenModel.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: TrainGenModelState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(TrainGenModelHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
        }

        html_map = {**html_map, **{
            'show_reports': any(len(rep_results) > 0 for rep_results in state.report_results.values()),
            'reports': list(itertools.chain.from_iterable(
                [Util.to_dict_recursive(Util.update_report_paths(report_result, base_path), base_path)
                 for report_result in state.report_results[report_type]]
                for report_type in state.report_results.keys()))
        }}

        return html_map
