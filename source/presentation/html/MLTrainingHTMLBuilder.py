import os

from source.IO.ml_method.MLExporter import MLExporter
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.ml_model_training.MLModelTrainingState import MLModelTrainingState


class MLTrainingHTMLBuilder:
    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"

    @staticmethod
    def build(state: MLModelTrainingState = None) -> str:
        base_path = PathBuilder.build(state.result_path + "../HTML_output/")
        html_map = MLTrainingHTMLBuilder.make_html_map(state, base_path)
        result_file = f"{base_path}MLModelTraining_{state.name}.html"
        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}MLTraining.html",
                             template_map=html_map, result_path=result_file)
        return result_file

    @staticmethod
    def make_html_map(state: MLModelTrainingState, base_path: str) -> dict:

        models_per_label = []

        for label, hp_item in state.hp_items.items():
            export_path = os.path.relpath(MLExporter.export(hp_item, f"{state.result_path}ml_export_{label}/"))
            filename = f"ml_model_{hp_item.hp_setting.ml_method_name}_{label}"
            model_zip_path = Util.make_downloadable_zip(state.result_path, export_path, filename)
            models_per_label.append({
                'label': label, 'model_zip_path': os.path.relpath(state.result_path + model_zip_path, base_path)
            })

        html_map = {'css_style': Util.get_css_content(MLTrainingHTMLBuilder.CSS_PATH), 'models_per_label': models_per_label,
                    'ml_reports': [{'label': label, 'reports': Util.to_dict_recursive(hp_item.model_report_results, base_path)}
                                   for label, hp_item in state.hp_items.items()],
                    'encoding_reports': [{'label': label, 'reports': Util.to_dict_recursive(hp_item.encoding_train_results, base_path)}
                                         for label, hp_item in state.hp_items.items()],
                    "show_ml_reports": any(len(hp_item.model_report_results) > 0 for hp_item in state.hp_items.values()),
                    "show_encoding_reports": any(len(hp_item.encoding_train_results) > 0 for hp_item in state.hp_items.values())}

        return html_map
