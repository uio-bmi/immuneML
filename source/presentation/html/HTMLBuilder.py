import os
from typing import List

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.InstructionPresentation import InstructionPresentation
from source.presentation.PresentationFactory import PresentationFactory
from source.presentation.PresentationFormat import PresentationFormat
from source.presentation.TemplateParser import TemplateParser


class HTMLBuilder:

    @staticmethod
    def build(states: list, path: str) -> str:
        presentations = HTMLBuilder._collect_all_presentations(states)
        rel_path = os.path.relpath(path) + '/'
        presentation_html_path = HTMLBuilder._make_document(presentations, rel_path)
        return presentation_html_path

    @staticmethod
    def _make_document(presentations: List[InstructionPresentation], path: str) -> str:
        result_path = f"{path}index.html"
        if len(presentations) > 1:
            html_map = {"instructions": presentations}
            TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}index.html",
                                 template_map=html_map, result_path=result_path)
        elif len(presentations) == 1:
            os.rename(presentations[0].path, result_path)
        else:
            result_path = None

        return result_path

    @staticmethod
    def _collect_all_presentations(states: list) -> List[InstructionPresentation]:
        presentations = []

        for state in states:
            presentation_builder = PresentationFactory.make_presentation_builder(state, PresentationFormat.HTML)
            presentation_path = presentation_builder.build(state, is_index=len(states) == 1)
            instruction_class = type(state).__name__[:-5]
            presentation = InstructionPresentation(presentation_path, instruction_class, state.name)
            presentations.append(presentation)

        return presentations
