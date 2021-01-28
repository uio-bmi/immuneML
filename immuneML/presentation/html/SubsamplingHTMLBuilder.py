from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper
from immuneML.workflows.instructions.subsampling.SubsamplingState import SubsamplingState


class SubsamplingHTMLBuilder:
    """
    A class that will make a HTML file(s) out of SubsamplingState object to show what analysis took place in
    the SubsamplingInstruction.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: SubsamplingState) -> str:
        """
        Function that builds the HTML files based on the Subsampling state.
        Arguments:
            state: SubsamplingState object including all details of the Subsampling instruction
        Returns:
             path to the main HTML file (which is located under state.result_path)
        """
        base_path = PathBuilder.build(state.result_path / "../HTML_output/")
        html_map = SubsamplingHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / f"Subsampling_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Subsampling.html",
                             template_map=html_map, result_path=result_file)

        return result_file

    @staticmethod
    def make_html_map(state: SubsamplingState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(SubsamplingHTMLBuilder.CSS_PATH),
            "name": state.name,
            'immuneML_version': MLUtil.get_immuneML_version(),
            "full_specs": Util.get_full_specs_path(base_path),
            "dataset_name": state.dataset.name if state.dataset.name is not None else state.dataset.identifier,
            "labels": [{"label_name": label} for label in state.dataset.get_label_names()],
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.dataset).__name__),
            "example_count": state.dataset.get_example_count(),
            "subsampled_datasets": [{"sub_dataset_iter": i,
                                     "sub_dataset_name": dataset.name,
                                     "dataset_size": f"{dataset.get_example_count()} {type(dataset).__name__.replace('Dataset', 's').lower()}",
                                     "formats": [{"dataset_download_link": item, "format_name": key}
                                                 for key, item in state.subsampled_dataset_paths[dataset.name].items()]}
                                    for i, dataset in enumerate(state.subsampled_datasets, 1)]
        }

        return html_map
