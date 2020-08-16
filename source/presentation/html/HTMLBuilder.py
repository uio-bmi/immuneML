import os
import shutil
from typing import List

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.InstructionPresentation import InstructionPresentation
from source.presentation.PresentationFactory import PresentationFactory
from source.presentation.PresentationFormat import PresentationFormat
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util


class HTMLBuilder:
    """
    Outputs HTML results of the analysis. This is currently the only defined format of presentation of results.

    Specification:

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
    def build(states: list, path: str) -> str:
        rel_path = os.path.relpath(path) + '/HTML_output/'
        presentations = HTMLBuilder._collect_all_presentations(states, rel_path)
        presentation_html_path = HTMLBuilder._make_document(presentations, rel_path)
        return presentation_html_path

    @staticmethod
    def _make_document(presentations: List[InstructionPresentation], path: str) -> str:
        result_path = f"{path}index.html"
        if len(presentations) > 1:
            html_map = {"instructions": presentations, "css_path": EnvironmentSettings.html_templates_path + "css/custom.css",
                        "full_specs": Util.get_full_specs_path(path)}
            TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}index.html",
                                 template_map=html_map, result_path=result_path)
        elif len(presentations) == 1:
            shutil.copyfile(presentations[0].path, result_path)
        else:
            result_path = None

        return result_path

    @staticmethod
    def _collect_all_presentations(states: list, rel_path: str) -> List[InstructionPresentation]:
        presentations = []

        for state in states:
            presentation_builder = PresentationFactory.make_presentation_builder(state, PresentationFormat.HTML)
            presentation_path = presentation_builder.build(state)
            if len(states) > 1:
                presentation_path = os.path.relpath(presentation_path, rel_path)
            instruction_class = type(state).__name__[:-5]
            presentation = InstructionPresentation(presentation_path, instruction_class, state.name)
            presentations.append(presentation)

        return presentations
