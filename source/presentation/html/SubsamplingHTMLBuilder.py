from source.environment.EnvironmentSettings import EnvironmentSettings
from source.ml_methods.util.Util import Util as MLUtil
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder
from source.util.StringHelper import StringHelper
from source.workflows.instructions.subsampling.SubsamplingState import SubsamplingState


class SubsamplingHTMLBuilder:
    """
    A class that will make a HTML file(s) out of SubsamplingState object to show what analysis took place in
    the SubsamplingInstruction.
    """

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(state: SubsamplingState) -> str:
        """
        Function that builds the HTML files based on the Subsampling state.
        Arguments:
            state: SubsamplingState object including all details of the Subsampling instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path + "../HTML_output/")
        html_map = SubsamplingHTMLBuilder.make_html_map(state, base_path)
        result_file = f"{base_path}Subsampling_{state.name}.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}Subsampling.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: SubsamplingState, base_path: str) -> dict:

        html_map = {
            "css_style": Util.get_css_content(SubsamplingHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "dataset_name": state.dataset.name if state.dataset.name is not None else state.dataset.identifier,
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.dataset).__name__),
            "example_count": state.dataset.get_example_count(),
            "subsampled_datasets": [{"sub_dataset_name": dataset.name, "dataset_size": dataset.get_example_count(),
                                     "formats": [{"dataset_download_link": item, "format_name": key}
                                                 for key, item in state.subsampled_dataset_paths[dataset.name].items()]}
                                    for dataset in state.subsampled_datasets]
        }

        return html_map
