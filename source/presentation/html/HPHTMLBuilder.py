import os

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.hyperparameter_optimization.states.HPOptimizationState import HPOptimizationState
from source.presentation.TemplateParser import TemplateParser
from source.presentation.html.Util import Util
from source.util.StringHelper import StringHelper


class HPHTMLBuilder:
    """
    A class that will make a HTML file(s) out of HPOptimizationState object to show what analysis took place in
    the HPOptimizationInstruction.
    """

    CSS_PATH = f"{EnvironmentSettings.html_templates_path}css/custom.css"
    NUM_DIGITS = 2

    @staticmethod
    def build(state: HPOptimizationState = None, is_index: bool = True) -> str:
        """
        Function that builds the HTML files based on the HPOptimization state.
        Arguments:
            state: HPOptimizationState object with all details on the optimization
            is_index: bool used to determine the paths in links depending on whether this is the only result of the app run or
                      there were multiple instructions
        Returns:
             path to the main HTML file (index.html which is located under state.result_path)
        """

        base_path = os.path.relpath(state.path + "/../") if is_index else os.path.relpath(state.path) + "/"
        html_map = HPHTMLBuilder.make_main_html_map(state, base_path)
        result_file = f"{state.path}HPOptimizationReport.html"

        TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}HPOptimization.html",
                             template_map=html_map, result_path=result_file)

        for index, item in enumerate(HPHTMLBuilder.make_assessment_indices_list(state, base_path)):
            TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}CVDetails.html",
                                 template_map=item, result_path=HPHTMLBuilder.make_assessment_split_path(index, state))

        for label in state.label_configuration.get_labels_by_name():
            for assessment_index in range(state.assessment.split_count):
                TemplateParser.parse(template_path=f"{EnvironmentSettings.html_templates_path}SelectionDetails.html",
                                     template_map=HPHTMLBuilder.make_selection(state, assessment_index, label, base_path),
                                     result_path=HPHTMLBuilder.make_selection_split_path(assessment_index, state, label))

        return result_file

    @staticmethod
    def make_assessment_split_path(split_index: int, state: HPOptimizationState) -> str:
        path = state.assessment_states[split_index].path + "CVDetails.html"
        return path

    @staticmethod
    def make_selection_split_path(assessment_index: int, state: HPOptimizationState, label: str):
        path = f"{state.assessment_states[assessment_index].path}selection_{state.selection.split_strategy.name.lower()}/" \
               f"{label}SelectionDetails.html"
        return path

    @staticmethod
    def make_selection(state: HPOptimizationState, assessment_index: int, label: str, base_path: str):
        selection_state = state.assessment_states[assessment_index].label_states[label].selection_state
        return {
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "label": label,
            "splits": [{"split_index": i} for i in range(1, state.selection.split_count + 1)],
            "split_count": state.selection.split_count,
            "optimization_metric": state.optimization_metric.name.lower(),
            "hp_settings": [{
                "hp_setting": hp_setting,
                "hp_splits": [{"optimization_metric_val": round(hp_item.performance, HPHTMLBuilder.NUM_DIGITS)}
                              if hp_item.performance is not None else "/" for hp_item in hp_items]
            } for hp_setting, hp_items in selection_state.hp_items.items()]
        }

    @staticmethod
    def make_assessment_indices_list(state: HPOptimizationState, base_path: str):
        return [{"css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
                 "optimization_metric": state.optimization_metric.name.lower(),
                 "split_index": assessment_state.split_index + 1,
                 "train_metadata_path": os.path.relpath(assessment_state.train_val_dataset.metadata_file, assessment_state.path),
                 "test_metadata_path": os.path.relpath(assessment_state.test_dataset.metadata_file, assessment_state.path),
                 "train_data_reports": Util.to_dict_recursive(assessment_state.train_val_data_reports, assessment_state.path),
                 "test_data_reports": Util.to_dict_recursive(assessment_state.test_data_reports, assessment_state.path),
                 "show_data_reports": len(assessment_state.train_val_data_reports) > 0 or len(
                     assessment_state.test_data_reports) > 0,
                 "labels": [{
                     "label": label,
                     "hp_settings": [{
                         "optimal": key == str(assessment_state.label_states[label].optimal_hp_setting),
                         "hp_setting": key,
                         "optimization_metric_val": round(item.performance, HPHTMLBuilder.NUM_DIGITS)
                     } for key, item in assessment_state.label_states[label].assessment_items.items()],
                     "selection_path": Util.get_relative_path(assessment_state.path,
                                                              HPHTMLBuilder.make_selection_split_path(i, state, label))
                 } for label in state.label_configuration.get_labels_by_name()],
                 }
                for i, assessment_state in enumerate(state.assessment_states)]

    @staticmethod
    def make_hp_per_label(state: HPOptimizationState, base_path: str):
        return [{"label": label, "assessment_results":
            [{"index": assessment_state.split_index + 1,
              "hp_setting": assessment_state.label_states[label].optimal_assessment_item.hp_setting,
              "optimization_metric_val": round(assessment_state.label_states[label].optimal_assessment_item.performance,
                                               HPHTMLBuilder.NUM_DIGITS),
              "split_details_path": Util.get_relative_path(base_path,
                                                           HPHTMLBuilder.make_assessment_split_path(assessment_state.split_index, state))}
             for i, assessment_state in enumerate(state.assessment_states)]} for label in
                state.label_configuration.get_labels_by_name()]

    @staticmethod
    def make_main_html_map(state: HPOptimizationState, base_path: str) -> dict:
        html_map = {
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
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
            "hp_per_label": HPHTMLBuilder.make_hp_per_label(state, base_path)
        }

        return html_map
