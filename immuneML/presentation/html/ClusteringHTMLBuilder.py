from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringState


class ClusteringHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: ClusteringState) -> Path:
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = ClusteringHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"DatasetExport_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Clustering.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state, base_path):
        return {}
