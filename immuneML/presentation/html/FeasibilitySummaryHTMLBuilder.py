from pathlib import Path
from typing import Union

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.ligo_sim_feasibility.FeasibilitySummaryInstruction import FeasibilitySummaryState


class FeasibilitySummaryHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: FeasibilitySummaryState) -> Path:
        """
        Function that builds the HTML files based on the ExploratoryAnalysis state.
        Arguments:
            state: ExploratoryAnalysisState object with details and results of the instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = FeasibilitySummaryHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"FeasibilitySummary_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "FeasibilitySummary.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: FeasibilitySummaryState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(FeasibilitySummaryHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            "sequence_count": state.sequence_count,
            "signal_names": ", ".join(s.id for s in state.signals),
            "gen_models": [{
                "name": name,
                "sig_freq": make_from_report_result(reports.signal_frequencies, base_path),
                "sig_coocc": make_from_report_result(reports.signal_cooccurrences, base_path),
                "p_gens": make_from_report_result(reports.p_gen_histogram, base_path),
                "seq_len": make_from_report_result(reports.seq_len_dist, base_path)
            } for name, reports in state.reports.items()]
        }

        return html_map


def make_from_report_result(report_result, base_path) -> Union[dict, None]:
    if report_result is not None:
        report_dict = Util.to_dict_recursive(Util.update_report_paths(report_result, base_path), base_path)
        report_dict['show_info'] = report_dict['info'] != '' and report_dict['info'] != None
        report_dict['show_text'] = len(report_dict['output_text']) > 0
        report_dict['show_tables'] = len(report_dict['output_tables']) > 0
        report_dict['show_figures'] = len(report_dict['output_figures']) > 0
        return report_dict
    else:
        return None
