from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder


class FailedGalaxyHTMLBuilder:
    """
    Constructs the HTML file to display for failed Galaxy runs
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(result_path, traceback_str) -> Path:
        html_map = {
            "css_style": Util.get_css_content(FailedGalaxyHTMLBuilder.CSS_PATH),
            "full_specs": "../full_specs.yaml",
            "logfile_content": open(result_path / "log.txt", 'r').read(),
            "exception": traceback_str,
            'immuneML_version': MLUtil.get_immuneML_version()}

        result_file = result_path / f"index.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "FailedGalaxy.html",
                             template_map=html_map, result_path=result_file)

        return result_file
