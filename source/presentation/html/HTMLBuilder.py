import os
import shutil
from typing import List
from pathlib import Path

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.util.Util import Util as MLUtil
from source.presentation.InstructionPresentation import InstructionPresentation
from source.presentation.PresentationFactory import PresentationFactory
from source.presentation.PresentationFormat import PresentationFormat
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util


class HTMLBuilder:
    """
    Outputs HTML results of the analysis. This is currently the only defined format of presentation of results.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ...
        instructions:
            ...
        output: # the output format
            format: HTML

    """

    @staticmethod
    def build(states: list, path: Path) -> Path:
        rel_path = Path(os.path.relpath(path))
        presentations = HTMLBuilder._collect_all_presentations(states, rel_path)
        presentation_html_path = HTMLBuilder._make_document(presentations, rel_path)
        return presentation_html_path

    @staticmethod
    def _make_document(presentations: List[InstructionPresentation], path: Path) -> Path:
        result_path = path / "index.html"
        if len(presentations) > 1:
            html_map = {"instructions": presentations, "css_path": EnvironmentSettings.html_templates_path / "css/custom.css",
                        "full_specs": Util.get_full_specs_path(path), 'immuneML_version': MLUtil.get_immuneML_version()}
            TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "index.html",
                                 template_map=html_map, result_path=result_path)
        elif len(presentations) == 1:
            shutil.copyfile(presentations[0].path, result_path)
            HTMLBuilder._update_paths(result_path)
        else:
            result_path = None

        return result_path

    @staticmethod
    def _update_paths(result_path: Path):
        with result_path.open('r') as file:

            lines = []
            for line in file.readlines():
                if "href=" in line and ".html" in line:
                    lines.append(line.split("href=\"")[0] + "href=\"./HTML_output/" + line.split("href=\"")[1])
                elif "src=" in line:
                    lines.append(line.split("src=\"")[0] + "src=\"./HTML_output/" + line.split("src=\"")[1])
                else:
                    lines.append(line)
                lines[-1] = lines[-1].replace("""href="../""", """href="./""")
                lines[-1] = lines[-1].replace("""src="../""", """src="./""")

        with result_path.open("w") as file:
            file.write("\n".join(lines))

    @staticmethod
    def _collect_all_presentations(states: list, rel_path: Path) -> List[InstructionPresentation]:
        presentations = []
        path = rel_path / 'HTML_output'

        for state in states:
            presentation_builder = PresentationFactory.make_presentation_builder(state, PresentationFormat.HTML)
            presentation_path = presentation_builder.build(state)
            if len(states) > 1:
                presentation_path = Path(os.path.relpath(presentation_path, path))
            instruction_class = type(state).__name__[:-5]
            presentation = InstructionPresentation(presentation_path, instruction_class, state.name)
            presentations.append(presentation)

        return presentations
