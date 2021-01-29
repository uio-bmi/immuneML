import os
from pathlib import Path

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper
from immuneML.workflows.instructions.ml_model_application.MLApplicationState import MLApplicationState


class MLApplicationHTMLBuilder:
    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"

    @staticmethod
    def build(state: MLApplicationState = None) -> str:
        base_path = PathBuilder.build(state.path / "../HTML_output/")
        html_map = MLApplicationHTMLBuilder.make_html_map(state, base_path)
        result_file = base_path / "MLModelTraining_{state.name}.html"
        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "MLApplication.html",
                             template_map=html_map, result_path=result_file)
        return result_file

    @staticmethod
    def make_html_map(state: MLApplicationState, base_path: Path) -> dict:
        return {
            "css_style": Util.get_css_content(MLApplicationHTMLBuilder.CSS_PATH),
            "hp_setting": state.hp_setting.get_key(),
            'immuneML_version': MLUtil.get_immuneML_version(),
            "label": state.label_config.get_labels_by_name()[0],
            "dataset_name": state.dataset.name,
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.dataset).__name__),
            "example_count": state.dataset.get_example_count(),
            "dataset_size": f"{state.dataset.get_example_count()} {type(state.dataset).__name__.replace('Dataset', 's').lower()}",
            "labels": [{"name": label, "values": str(state.label_config.get_label_values(label))[1:-1]}
                       for label in state.label_config.get_labels_by_name()],
            "predictions": Util.get_table_string_from_csv(state.predictions_path),
            "predictions_download_link": os.path.relpath(state.predictions_path, base_path)
        }
