import os
import statistics

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.PathBuilder import PathBuilder
from source.util.StringHelper import StringHelper


class HPHTMLBuilder:
    """
    A class that will make HTML file(s) out of HPOptimizationState object to show what analysis took place in
    the TrainMLModel.
    """

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"
    NUM_DIGITS = 2

    @staticmethod
    def build(state: HPOptimizationState = None) -> str:
        """
        Function that builds the HTML files based on the HPOptimization state.
        Arguments:
            state: HPOptimizationState object with all details on the optimization
        Returns:
             path to the main HTML file (index.html which is located under state.result_path)
        """

        base_path = PathBuilder.build(state.path + "../HTML_output/")
        html_map = HPHTMLBuilder.make_main_html_map(state, base_path)
        result_file = f"{base_path}TrainMLModelReport_{state.name}.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}HPOptimization.html",
                             template_map=html_map, result_path=result_file)

        for label in state.label_configuration.get_labels_by_name():
            for index, item in enumerate(HPHTMLBuilder.make_assessment_pages(state, base_path, label)):
                TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}AssessmentSplitDetails.html",
                                     template_map=item,
                                     result_path=f"{base_path}{HPHTMLBuilder.make_assessment_split_path(index, state.name, label)}")

        for label in state.label_configuration.get_labels_by_name():
            for assessment_index in range(state.assessment.split_count):
                TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}SelectionDetails.html",
                                     template_map=HPHTMLBuilder.make_selection(state, assessment_index, label, base_path),
                                     result_path=f"{base_path}{HPHTMLBuilder.make_selection_split_path(assessment_index, label, state.name)}")

        return result_file

    @staticmethod
    def make_assessment_split_path(split_index: int, state_name: str, label: str) -> str:
        return f"{state_name}_CV_assessment_split_{split_index + 1}_{label}.html"

    @staticmethod
    def make_selection_split_path(assessment_index: int, label: str, state_name: str):
        return f"{state_name}_selection_details_{label}_split_{assessment_index + 1}.html"

    @staticmethod
    def make_selection(state: HPOptimizationState, assessment_index: int, label: str, base_path):
        selection_state = state.assessment_states[assessment_index].label_states[label].selection_state

        hp_settings = []
        optimal = selection_state.optimal_hp_setting.get_key()

        for hp_setting, hp_items in selection_state.hp_items.items():
            hp_settings.append({
                "hp_setting": hp_setting,
                "hp_splits": [{"optimization_metric_val": round(hp_item.performance, HPHTMLBuilder.NUM_DIGITS)} if hp_item.performance is not None else "/" for hp_item in hp_items],
                "optimal": hp_setting == optimal
            })

            performances = [round(hp_item.performance, HPHTMLBuilder.NUM_DIGITS) for hp_item in hp_items if hp_item.performance is not None]
            if len(performances) > 1:
                hp_settings[-1]["average"] = round(statistics.mean(performances), HPHTMLBuilder.NUM_DIGITS)
                hp_settings[-1]["show_average"] = True
            else:
                hp_settings[-1]["average"] = None
                hp_settings[-1]["show_average"] = False

        return {
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "label": label,
            "assessment_split": assessment_index + 1,
            "splits": [{"split_index": i} for i in range(1, state.selection.split_count + 1)],
            "split_count": state.selection.split_count,
            "optimization_metric": state.optimization_metric.name.lower(),
            "hp_settings": hp_settings,
            "show_average": any(hps["show_average"] for hps in hp_settings),
            "data_split_reports": [
                {'split_index': index + 1,
                 'train': Util.to_dict_recursive(selection_state.train_data_reports[index], base_path)
                 if len(selection_state.train_data_reports) == state.selection.split_count else None,
                 'test': Util.to_dict_recursive(selection_state.val_data_reports[index], base_path)
                 if len(selection_state.train_data_reports) == state.selection.split_count else None}
                for index in range(state.selection.split_count)] if len(state.selection.reports.data_split_reports) > 0 else None,
            "has_data_split_reports": len(state.selection.reports.data_split_reports) > 0,
            "has_reports_per_setting": len(state.selection.reports.encoding_reports) + len(state.selection.reports.model_reports) > 0,
            "reports_per_setting": [{
                "hp_setting": hp_setting,
                "reports": HPHTMLBuilder.make_selection_reports_for_item_list(hp_items, base_path)
            } for hp_setting, hp_items in selection_state.hp_items.items()]
        }

    @staticmethod
    def make_selection_reports_for_item_list(hp_items: list, base_path) -> list:
        result = []

        for split_index, hp_item in enumerate(hp_items):
            result.append({
                "split_index": split_index + 1,
                "encoding_train_reports": Util.to_dict_recursive(hp_item.encoding_train_results, base_path) if len(
                    hp_item.encoding_train_results) > 0 else None,
                "encoding_test_reports": Util.to_dict_recursive(hp_item.encoding_test_results, base_path) if len(
                    hp_item.encoding_test_results) > 0 else None,
                "ml_reports": Util.to_dict_recursive(hp_item.model_report_results, base_path) if len(hp_item.model_report_results) > 0 else None,
            })

        return result if len(result) > 0 else None

    @staticmethod
    def make_assessment_pages(state: HPOptimizationState, base_path: str, label: str):

        assessment_list = []

        for i, assessment_state in enumerate(state.assessment_states):

            assessment_item = {"css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
                               "optimization_metric": state.optimization_metric.name.lower(),
                               "split_index": assessment_state.split_index + 1,
                               "hp_settings": [],
                               "has_reports": len(state.assessment.reports.model_reports) + len(state.assessment.reports.encoding_reports) > 0,
                               "train_data_reports": Util.to_dict_recursive(assessment_state.train_val_data_reports, base_path),
                               "test_data_reports": Util.to_dict_recursive(assessment_state.test_data_reports, base_path),
                               "show_data_reports": len(assessment_state.train_val_data_reports) > 0 or len(assessment_state.test_data_reports) > 0}

            if hasattr(assessment_state.train_val_dataset, "metadata_file") and assessment_state.train_val_dataset.metadata_file is not None:
                assessment_item["train_metadata_path"] = os.path.relpath(assessment_state.train_val_dataset.metadata_file, base_path)
                assessment_item["train_metadata"] = Util.get_table_string_from_csv(assessment_state.train_val_dataset.metadata_file)
            else:
                assessment_item["train_metadata_path"] = None

            if hasattr(assessment_state.test_dataset, "metadata_file") and assessment_state.test_dataset.metadata_file is not None:
                assessment_item['test_metadata_path'] = os.path.relpath(assessment_state.test_dataset.metadata_file, base_path)
                assessment_item["test_metadata"] = Util.get_table_string_from_csv(assessment_state.test_dataset.metadata_file)
            else:
                assessment_item["test_metadata_path"] = None

            assessment_item["label"] = label
            for hp_setting, item in assessment_state.label_states[label].assessment_items.items():
                optimal = str(assessment_state.label_states[label].optimal_hp_setting.get_key())
                assessment_item["hp_settings"].append({
                    "optimal": hp_setting.get_key() == optimal,
                    "hp_setting": hp_setting.get_key(),
                    "optimization_metric_val": round(item.performance, HPHTMLBuilder.NUM_DIGITS),
                    "reports_path": HPHTMLBuilder.make_assessment_reports(state, i, hp_setting, assessment_state, label, base_path)
                })

            assessment_item["selection_path"] = HPHTMLBuilder.make_selection_split_path(i, label, state.name)

            assessment_list.append(assessment_item)

        return assessment_list

    @staticmethod
    def make_assessment_reports(state, i, hp_setting_key, assessment_state, label, base_path: str):
        path = f"{base_path}{state.name}_{label}_{hp_setting_key}_assessment_reports_split_{i + 1}.html"

        hp_item = assessment_state.label_states[label].assessment_items[hp_setting_key]
        data = {
            "split_index": i + 1,
            "hp_setting": hp_setting_key,
            "label": label,
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "encoding_train_reports": Util.to_dict_recursive(hp_item.encoding_train_results, base_path) if len(
                hp_item.encoding_train_results) > 0 else None,
            "encoding_test_reports": Util.to_dict_recursive(hp_item.encoding_test_results, base_path) if len(
                hp_item.encoding_test_results) > 0 else None,
            "ml_reports": Util.to_dict_recursive(hp_item.model_report_results, base_path) if len(
                hp_item.model_report_results) > 0 else None
        }

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}Reports.html", template_map=data,
                             result_path=path)

        return os.path.basename(path)

    @staticmethod
    def make_hp_per_label(state: HPOptimizationState):
        mapping = []

        for label in state.label_configuration.get_labels_by_name():
            results = []
            for i, assessment_state in enumerate(state.assessment_states):
                results.append({
                    "index": assessment_state.split_index + 1,
                    "hp_setting": assessment_state.label_states[label].optimal_assessment_item.hp_setting,
                    "optimization_metric_val": round(assessment_state.label_states[label].optimal_assessment_item.performance,
                                                     HPHTMLBuilder.NUM_DIGITS),
                    "split_details_path": HPHTMLBuilder.make_assessment_split_path(assessment_state.split_index, state.name, label)
                })

            mapping.append({"label": label, "assessment_results": results})

        return mapping

    @staticmethod
    def make_model_per_label(state: HPOptimizationState, base_path: str) -> list:

        mapping = []

        for label in state.label_configuration.get_labels_by_name():
            mapping.append({
                "label": label,
                "model_path": os.path.relpath(path=state.optimal_hp_item_paths[label], start=base_path)
            })

        return mapping

    @staticmethod
    def make_main_html_map(state: HPOptimizationState, base_path: str) -> dict:
        html_map = {
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            "dataset_name": state.dataset.name if state.dataset.name is not None else state.dataset.identifier,
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.dataset).__name__),
            "example_count": state.dataset.get_example_count(),
            "labels": [{"name": label.name, "values": str(label.values)[1:-1]} for label in state.label_configuration.get_label_objects()],
            "optimization_metric": state.optimization_metric.name.lower(),
            "other_metrics": str([metric.name.lower() for metric in state.metrics])[1:-1].replace("'", ""),
            "metrics": [{"name": metric.name.lower()} for metric in state.metrics],
            "assessment_desc": state.assessment,
            "selection_desc": state.selection,
            "dataset_reports": Util.to_dict_recursive(state.data_report_results, base_path) if state.data_report_results else None,
            "show_dataset_reports": bool(state.data_report_results),
            "show_hp_reports": bool(state.hp_report_results),
            'hp_reports': Util.to_dict_recursive(state.hp_report_results, base_path) if state.hp_report_results else None,
            "hp_per_label": HPHTMLBuilder.make_hp_per_label(state),
            'models_per_label': HPHTMLBuilder.make_model_per_label(state, base_path)
        }

        return html_map
