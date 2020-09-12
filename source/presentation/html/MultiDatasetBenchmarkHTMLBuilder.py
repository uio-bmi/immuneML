import os

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util


class MultiDatasetBenchmarkHTMLBuilder:

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(report_results: dict, result_path: str, instruction_result_paths: dict) -> str:
        html_map = MultiDatasetBenchmarkHTMLBuilder._make_html_map(report_results, result_path, instruction_result_paths)
        result_file = f"{result_path}index.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}MultiDatasetBenchmark.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def _make_html_map(report_results: dict, result_path: str, instruction_result_paths: dict) -> dict:
        html_map = {
            "css_style": Util.get_css_content(MultiDatasetBenchmarkHTMLBuilder.CSS_PATH),
            "reports": Util.to_dict_recursive(report_results.values(), result_path),
            "show_reports": True,
            "instruction_overviews": [{"name": name, "path": os.path.relpath(path + "index.html", result_path)}
                                      for name, path in instruction_result_paths.items()]
        }

        if len(html_map['reports']) == 0:
            html_map['show_reports'] = False

        return html_map
