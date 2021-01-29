import io
import os
import statistics
from pathlib import Path

import pandas as pd

from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Metric import Metric
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.hyperparameter_optimization.states.HPAssessmentState import HPAssessmentState
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.HPLabelState import HPLabelState
from immuneML.hyperparameter_optimization.states.HPSelectionState import HPSelectionState
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_methods.util.Util import Util as MLUtil
from immuneML.presentation.TemplateParser import TemplateParser
from immuneML.presentation.html.Util import Util
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper


class HPHTMLBuilder:
    """
    A class that will make HTML file(s) out of TrainMLModelState object to show what analysis took place in the TrainMLModel.
    """

    CSS_PATH = EnvironmentSettings.html_templates_path / "css/custom.css"
    NUM_DIGITS = 3

    @staticmethod
    def build(state: TrainMLModelState = None) -> Path:
        """
        Function that builds the HTML files based on the HPOptimization state.
        Arguments:
            state: HPOptimizationState object with all details on the optimization
        Returns:
             path to the main HTML file (index.html which is located under state.result_path)
        """

        base_path = PathBuilder.build(state.path / "../HTML_output/")
        state = HPHTMLBuilder._move_reports_recursive(state, base_path)
        html_map = HPHTMLBuilder._make_main_html_map(state, base_path)
        result_file = base_path / f"TrainMLModelReport_{state.name}.html"

        TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "HPOptimization.html",
                             template_map=html_map, result_path=result_file)

        for label in state.label_configuration.get_labels_by_name():
            for index, item in enumerate(HPHTMLBuilder._make_assessment_pages(state, base_path, label)):
                TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "AssessmentSplitDetails.html",
                                     template_map=item,
                                     result_path=base_path / HPHTMLBuilder._make_assessment_split_path(index, state.name, label))

        for label in state.label_configuration.get_labels_by_name():
            for assessment_index in range(state.assessment.split_count):
                TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "SelectionDetails.html",
                                     template_map=HPHTMLBuilder._make_selection(state, assessment_index, label, base_path),
                                     result_path=base_path / HPHTMLBuilder._make_selection_split_path(assessment_index, label, state.name))

        return result_file

    @staticmethod
    def _make_assessment_split_path(split_index: int, state_name: str, label: str) -> Path:
        return Path(f"{state_name}_CV_assessment_split_{split_index + 1}_{label}.html")

    @staticmethod
    def _make_selection_split_path(assessment_index: int, label: str, state_name: str) -> Path:
        return Path(f"{state_name}_selection_details_{label}_split_{assessment_index + 1}.html")

    @staticmethod
    def _make_selection(state: TrainMLModelState, assessment_index: int, label: str, base_path):
        selection_state = state.assessment_states[assessment_index].label_states[label].selection_state

        hp_settings = []
        optimal = selection_state.optimal_hp_setting.get_key()

        for hp_setting, hp_items in selection_state.hp_items.items():
            hp_splits = []
            for hp_item in hp_items:
                hp_splits.append(HPHTMLBuilder._print_metric(hp_item.performance, state.optimization_metric))
            hp_settings.append({
                "hp_setting": hp_setting,
                "hp_splits": hp_splits,
                "optimal": hp_setting == optimal
            })

            performances = [HPHTMLBuilder._print_metric(hp_item.performance, state.optimization_metric) for hp_item in hp_items]
            if len(performances) > 1:
                hp_settings[-1]["average"] = round(statistics.mean(perf for perf in performances if [isinstance(perf, float)]), HPHTMLBuilder.NUM_DIGITS)
                hp_settings[-1]["show_average"] = True
            else:
                hp_settings[-1]["average"] = None
                hp_settings[-1]["show_average"] = False

        has_other_metrics = len([metric for metric in state.metrics if metric != state.optimization_metric]) > 0 and \
                            not (state.selection.split_strategy == SplitType.RANDOM and state.selection.training_percentage == 1)

        return {
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "label": label,
            "assessment_split": assessment_index + 1,
            "splits": [{"split_index": i} for i in range(1, state.selection.split_count + 1)],
            "split_count": state.selection.split_count,
            "optimization_metric": state.optimization_metric.name.lower(),
            "has_other_metrics": has_other_metrics,
            "metrics": [{"performance": HPHTMLBuilder._extract_selection_performance_per_metric(selection_state, metric, state.selection.split_count),
                         "metric": HPHTMLBuilder._get_heading_metric_name(metric.name.lower())}
                        for metric in state.metrics if metric != state.optimization_metric] if has_other_metrics else None,
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
                "reports": HPHTMLBuilder._make_selection_reports_for_item_list(hp_items, base_path)
            } for hp_setting, hp_items in selection_state.hp_items.items()]
        }

    @staticmethod
    def _make_selection_reports_for_item_list(hp_items: list, base_path) -> list:
        result = []

        for split_index, hp_item in enumerate(hp_items):
            result.append({
                "split_index": split_index + 1,
                "has_encoding_train_reports": len(hp_item.encoding_train_results) > 0,
                "has_encoding_test_reports": len(hp_item.encoding_test_results) > 0,
                "has_ml_reports": len(hp_item.model_report_results) > 0,
                "encoding_train_reports": Util.to_dict_recursive(hp_item.encoding_train_results, base_path) if len(
                    hp_item.encoding_train_results) > 0 else None,
                "encoding_test_reports": Util.to_dict_recursive(hp_item.encoding_test_results, base_path) if len(
                    hp_item.encoding_test_results) > 0 else None,
                "ml_reports": Util.to_dict_recursive(hp_item.model_report_results, base_path) if len(hp_item.model_report_results) > 0 else None,
            })

        return result if len(result) > 0 else None

    @staticmethod
    def _make_assessment_pages(state: TrainMLModelState, base_path: Path, label: str):
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
                assessment_item["train_metadata_path"] = os.path.relpath(str(assessment_state.train_val_dataset.metadata_file), str(base_path))
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
                reports_path = HPHTMLBuilder._make_assessment_reports(state, i, hp_setting, assessment_state, label, base_path)
                assessment_item["hp_settings"].append({
                    "optimal": str(hp_setting) == optimal,
                    "hp_setting": str(hp_setting),
                    "optimization_metric_val": HPHTMLBuilder._print_metric(item.performance, state.optimization_metric),
                    "reports_path": reports_path
                })
            assessment_item["show_non_optimal"] = len(assessment_item["hp_settings"]) > 1

            assessment_item["selection_path"] = HPHTMLBuilder._make_selection_split_path(i, label, state.name)
            assessment_item['performances_per_metric'] = HPHTMLBuilder._extract_assessment_performances_per_metric(state, assessment_state, label)

            assessment_list.append(assessment_item)

        return assessment_list

    @staticmethod
    def _extract_assessment_performances_per_metric(state: TrainMLModelState, assessment_state: HPAssessmentState, label: str) -> str:
        performance_metric = {"setting": [], **{metric.name.lower(): [] for metric in state.metrics}}
        for hp_setting, hp_item in assessment_state.label_states[label].assessment_items.items():
            performance_metric['setting'].append(str(hp_setting))
            for metric in sorted(state.metrics, key=lambda metric: metric.name.lower()):
                performance_metric[metric.name.lower()].append(HPHTMLBuilder._print_metric(hp_item.performance, metric))

        s = io.StringIO()
        pd.DataFrame(performance_metric).rename(columns={"setting": 'Hyperparameter settings (preprocessing, encoding, ML method)'})\
            .to_csv(s, sep="\t", index=False)
        return Util.get_table_string_from_csv_string(s.getvalue(), separator="\t")

    @staticmethod
    def _make_assessment_reports(state, i, hp_setting_key, assessment_state, label, base_path: Path):
        path = base_path / f"{state.name}_{label}_{hp_setting_key}_assessment_reports_split_{i + 1}.html"

        hp_item = assessment_state.label_states[label].assessment_items[hp_setting_key]
        data = {
            "split_index": i + 1,
            "hp_setting": hp_setting_key,
            "label": label,
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "has_encoding_reports": len(hp_item.encoding_train_results) > 0 or len(hp_item.encoding_test_results) > 0,
            "has_ml_reports": len(hp_item.model_report_results) > 0,
            "encoding_train_reports": Util.to_dict_recursive(hp_item.encoding_train_results, base_path) if len(
                hp_item.encoding_train_results) > 0 else None,
            "encoding_test_reports": Util.to_dict_recursive(hp_item.encoding_test_results, base_path) if len(
                hp_item.encoding_test_results) > 0 else None,
            "ml_reports": Util.to_dict_recursive(hp_item.model_report_results, base_path) if len(
                hp_item.model_report_results) > 0 else None
        }

        if data["has_ml_reports"] or data["has_encoding_reports"]:
            TemplateParser.parse(template_path=EnvironmentSettings.html_templates_path / "Reports.html", template_map=data, result_path=path)
            return path.name
        else:
            return None

    @staticmethod
    def _make_hp_per_label(state: TrainMLModelState):
        mapping = []

        for label in state.label_configuration.get_labels_by_name():
            results = []
            for i, assessment_state in enumerate(state.assessment_states):
                results.append({
                    "index": assessment_state.split_index + 1,
                    "hp_setting": assessment_state.label_states[label].optimal_assessment_item.hp_setting,
                    "optimization_metric_val": HPHTMLBuilder._print_metric(assessment_state.label_states[label].optimal_assessment_item.performance,
                                                                           state.optimization_metric),
                    "split_details_path": HPHTMLBuilder._make_assessment_split_path(assessment_state.split_index, state.name, label)
                })

            mapping.append({"label": label, "assessment_results": results})

        return mapping

    @staticmethod
    def _print_metric(performance: dict, metric: Metric):
        if performance is not None and metric.name.lower() in performance:
            if isinstance(performance[metric.name.lower()], float):
                return round(performance[metric.name.lower()], HPHTMLBuilder.NUM_DIGITS)
            else:
                return performance[metric.name.lower()]
        else:
            return Constants.NOT_COMPUTED

    @staticmethod
    def _make_model_per_label(state: TrainMLModelState, base_path: Path) -> list:

        mapping = []

        for label in state.label_configuration.get_labels_by_name():
            mapping.append({
                "label": label,
                "model_path": Path(os.path.relpath(path=str(state.optimal_hp_item_paths[label]), start=str(base_path)))
            })

        return mapping

    @staticmethod
    def _make_main_html_map(state: TrainMLModelState, base_path: Path) -> dict:
        html_map = {
            "css_style": Util.get_css_content(HPHTMLBuilder.CSS_PATH),
            "full_specs": Util.get_full_specs_path(base_path),
            "dataset_name": state.dataset.name if state.dataset.name is not None else state.dataset.identifier,
            "dataset_type": StringHelper.camel_case_to_word_string(type(state.dataset).__name__),
            "example_count": state.dataset.get_example_count(),
            "dataset_size": f"{state.dataset.get_example_count()} {type(state.dataset).__name__.replace('Dataset', 's').lower()}",
            "labels": [{"name": label.name, "values": str(label.values)[1:-1]} for label in state.label_configuration.get_label_objects()],
            "optimization_metric": state.optimization_metric.name.lower(),
            "other_metrics": str([metric.name.lower() for metric in state.metrics])[1:-1].replace("'", ""),
            "metrics": [{"name": metric.name.lower()} for metric in state.metrics],
            "assessment_desc": state.assessment,
            "selection_desc": state.selection,
            "show_hp_reports": bool(state.report_results),
            'hp_reports': Util.to_dict_recursive(state.report_results, base_path) if state.report_results else None,
            "hp_per_label": HPHTMLBuilder._make_hp_per_label(state),
            'models_per_label': HPHTMLBuilder._make_model_per_label(state, base_path),
            'immuneML_version': MLUtil.get_immuneML_version()
        }

        return html_map

    @staticmethod
    def _move_reports_recursive(obj, path: Path):
        for attribute in (vars(obj) if not isinstance(obj, dict) else obj):
            attribute_value = getattr(obj, attribute) if not isinstance(obj, dict) else obj[attribute]
            if isinstance(attribute_value, list) and all(isinstance(item, ReportResult) for item in attribute_value):
                new_attribute_values = []
                for report_result in attribute_value:
                    new_attribute_values.append(Util.update_report_paths(report_result, path))
                setattr(obj, attribute, new_attribute_values)
            elif isinstance(attribute_value, list) and all(isinstance(item, HPAssessmentState) for item in attribute_value):
                obj = HPHTMLBuilder._process_list_recursively(obj, attribute, attribute_value, path)
            elif isinstance(attribute_value, dict) and all(
                    isinstance(item, HPLabelState) or isinstance(item, HPItem) for item in attribute_value.values()):
                obj = HPHTMLBuilder._process_dict_recursive(obj, attribute, attribute_value, path)
            elif isinstance(attribute_value, dict) and all(isinstance(item, list) for item in attribute_value.values()) and all(
                    all(isinstance(item, HPItem) for item in item_list) for item_list in attribute_value.values()):
                obj = HPHTMLBuilder._process_hp_items(obj, attribute, attribute_value, path)
            elif isinstance(attribute_value, HPSelectionState):
                setattr(obj, attribute, HPHTMLBuilder._move_reports_recursive(attribute_value, path))

        return obj

    @staticmethod
    def _process_hp_items(obj, attribute, attribute_value, path: Path):
        new_attribute_value = {}
        for hp_setting, hp_item_list in attribute_value.items():
            new_hp_item_list = []
            for hp_item in hp_item_list:
                new_hp_item_list.append(HPHTMLBuilder._move_reports_recursive(hp_item, path))
            new_attribute_value[hp_setting] = new_hp_item_list

        setattr(obj, attribute, new_attribute_value)

        return obj

    @staticmethod
    def _process_dict_recursive(obj, attribute, attribute_value, path: Path):
        for key, value in attribute_value.items():
            attribute_value[key] = HPHTMLBuilder._move_reports_recursive(value, path)
        setattr(obj, attribute, attribute_value)
        return obj

    @staticmethod
    def _process_list_recursively(obj, attribute, attribute_value, path: Path):
        new_attribute_values = []
        for item in attribute_value:
            new_attribute_values.append(HPHTMLBuilder._move_reports_recursive(item, path))
        setattr(obj, attribute, new_attribute_values)
        return obj

    @staticmethod
    def _extract_selection_performance_per_metric(selection_state: HPSelectionState, metric: Metric, split_count):
        performance = {"setting": [], **{f"split {i + 1}": [] for i in range(split_count)}}
        for hp_setting, hp_item_list in selection_state.hp_items.items():
            performance['setting'].append(str(hp_setting))
            for index, hp_item in enumerate(hp_item_list):
                performance[f'split {index + 1}'].append(HPHTMLBuilder._print_metric(hp_item.performance, metric))

        s = io.StringIO()
        pd.DataFrame(performance).rename(columns={"setting": 'Hyperparameter settings (preprocessing, encoding, ML method)'}).to_csv(s, sep="\t",
                                                                                                                                     index=False)
        return Util.get_table_string_from_csv_string(s.getvalue(), separator="\t")

    @staticmethod
    def _get_heading_metric_name(metric: str):
        if metric != "auc":
            return " ".join(metric.split("_")).title()
        else:
            return metric.upper()
