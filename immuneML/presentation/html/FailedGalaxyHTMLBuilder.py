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
    def build(result_path) -> Path:
        base_path = PathBuilder.build(result_path / "../HTML_output/")
        html_map = {
            "css_style": Util.get_css_content(FailedGalaxyHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            "logfile": Util.get_logfile_path(base_path),
            'immuneML_version': MLUtil.get_immuneML_version()}

        result_file = base_path / f"index.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "FailedGalaxy.html",
                             template_map=html_map, result_path=result_file)

        return result_file
