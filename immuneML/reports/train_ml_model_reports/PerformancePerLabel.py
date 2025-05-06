import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class PerformancePerLabel(TrainMLModelReport):
    """
    Report that shows the performance of the model where the examples are grouped by alternative_label. It can be used
    to investigate if the model is learning the alternative_label instead of label of interest for classification.

    **Specification arguments:**

    - alternative_label (str): The name of the alternative_label column in the dataset.

    - metric (str): The metric to use for the report. Default is balanced_accuracy.

    - compute_for_selection (bool): If True, the report will be computed for the selection. Default is True.

    - compute_for_assessment (bool): If True, the report will be computed for the assessment. Default is True.

    - plot_on_test (bool): If True, the report will be plotted on the test data. Default is True.

    - plot_on_train (bool): If True, the report will be plotted on the training data. Default is False.

    **YAML specification:**

    .. code-block:: yaml

        reports:
            my_report:
                PerformancePerLabel:
                    alternative_label: batch
                    metric: balanced_accuracy

    """

    def __init__(self, alternative_label: str, metric: str = "balanced_accuracy", compute_for_selection: bool = True,
                 compute_for_assessment: bool = True, state: TrainMLModelState = None, result_path: Path = None,
                 name: str = None, label: Label = None, number_of_processes: int = 1, plot_on_train: bool = False,
                 plot_on_test: bool = True):
        super().__init__(name=name, state=state, label=label, result_path=result_path,
                         number_of_processes=number_of_processes)
        self.alternative_label = alternative_label
        self.alternative_label_values = []
        self.metric = metric.lower()
        self.compute_for_selection = compute_for_selection
        self.compute_for_assessment = compute_for_assessment
        self.plot_on_train = plot_on_train
        self.plot_on_test = plot_on_test

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()),
                                               ["alternative_label", 'compute_for_selection', 'compute_for_assessment',
                                                'metric'], "PerformancePerLabel",
                                               "PerformancePerLabel")

        alternative_label = kwargs["alternative_label"]
        metric = kwargs["metric"] if "metric" in kwargs else "balanced_accuracy"
        compute_for_selection = kwargs["compute_for_selection"]
        compute_for_assessment = kwargs["compute_for_assessment"]
        name = kwargs["name"] if "name" in kwargs else None

        return PerformancePerLabel(alternative_label=alternative_label, metric=metric,
                                   compute_for_selection=compute_for_selection,
                                   compute_for_assessment=compute_for_assessment,
                                   name=name)

    def discover_alternative_label_values(self):
        self.alternative_label_values = sorted(self.state.dataset.get_metadata([self.alternative_label], return_df=True)[
            self.alternative_label].unique().tolist())

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        report_outputs = []

        self.discover_alternative_label_values()

        if self.compute_for_selection:
            selection_outputs = self._generate_for_selection()
            report_outputs.extend(selection_outputs)

        if self.compute_for_assessment:
            assessment_outputs = self._generate_for_assessment()
            report_outputs.extend(assessment_outputs)

        return ReportResult(name=self.name,
                            info="Performance per label report showing how the model performs on different subgroups of data.",
                            output_tables=[output for output in report_outputs if output.path.suffix == '.csv'],
                            output_figures=[output for output in report_outputs if output.path.suffix == '.html'])

    def _generate_for_selection(self):
        outputs = []
        for assessment_index, assessment_state in enumerate(self.state.assessment_states):
            for label_name, label_state in assessment_state.label_states.items():
                data = self._get_performance_data(label_state.selection_state.hp_items)
                outputs.extend(self._write_split_results(data, f"{label_name}_assessment_{assessment_index+1}_selection", label_name))
        return outputs

    def _generate_for_assessment(self):
        outputs = []
        for label in self.state.label_configuration.get_label_objects():
            assessment_items_per_setting = {hp_setting.get_key(): [] for hp_setting in self.state.hp_settings}

            for assessment_index, assessment_state in enumerate(self.state.assessment_states):
                for hp_setting in self.state.hp_settings:
                    assessment_items_per_setting[hp_setting.get_key()].append(assessment_state.label_states[label.name].assessment_items[hp_setting.get_key()])

            data = self._get_performance_data(assessment_items_per_setting)

            outputs.extend(self._write_split_results(data, f"{label.name}_assessment", label.name))
        return outputs

    def _get_performance_for_dataset(self, predictions: pd.DataFrame, dataset: Dataset, label: Label, counts: dict):
        data = []

        for label_value in self.alternative_label_values:
            mask = dataset.get_metadata([self.alternative_label], return_df=True)[self.alternative_label] == label_value
            indices = np.where(mask)[0]
            if len(indices) > 0:
                true = predictions[f"{label.name}_true_class"].iloc[indices]
                predicted = predictions[f"{label.name}_predicted_class"].iloc[indices]
                if f"{label.name}_{label.positive_class}_proba" in predictions.columns:
                    proba = predictions[f"{label.name}_{label.positive_class}_proba"].iloc[indices]
                else:
                    proba = None
                performance = MetricUtil.score_for_metric(
                    metric=ClassificationMetric[self.metric.upper()],
                    true_y=true,
                    predicted_y=predicted,
                    predicted_proba_y=proba,
                    classes=label.values
                )
            else:
                performance = float('nan')

            data.append({
                "alternative_label_value": label_value,
                "count": counts.get(label_value, 0),
                "performance": performance
            })

        return data

    def _get_performance_data(self, hp_items: Dict[str, List[HPItem]]):
        dfs = {'train': [], 'test': []}

        for label in self.state.label_configuration.get_label_objects():
            for setting_name, hp_item_results in hp_items.items():

                for run_id, hp_item_result in enumerate(hp_item_results):

                    datasets = self._prepare_datasets(hp_item_result)

                    for dataset_info in datasets:
                        if (dataset_info['desc'] == 'train' and self.plot_on_train) or (dataset_info['desc'] == 'test' and self.plot_on_test):
                            data = self._process_dataset(dataset_info, label, hp_item_result)
                            dfs[dataset_info['desc']].append({**data, 'run_id': run_id+1})

        return {'train': pd.DataFrame(dfs['train']) if self.plot_on_train else None,
                'test': pd.DataFrame(dfs['test'] if self.plot_on_test else [])}

    def _prepare_datasets(self, hp_item: HPItem):
        train_metadata = hp_item.train_dataset.get_metadata([self.alternative_label], return_df=True)
        test_metadata = hp_item.test_dataset.get_metadata([self.alternative_label], return_df=True)
        
        train_counts = train_metadata[self.alternative_label].value_counts()
        test_counts = test_metadata[self.alternative_label].value_counts()

        train_predictions = self._load_predictions(hp_item.train_predictions_path)
        test_predictions = self._load_predictions(hp_item.test_predictions_path)

        if train_predictions is None or test_predictions is None:
            self._log_missing_predictions(hp_item)
            return None

        return [
            {'desc': 'train', 'dataset': hp_item.train_dataset, 'predictions': train_predictions, 
             'counts': train_counts, 'metadata': train_metadata},
            {'desc': 'test', 'dataset': hp_item.test_dataset, 'predictions': test_predictions, 
             'counts': test_counts, 'metadata': test_metadata}
        ]

    def _load_predictions(self, path):
        return pd.read_csv(path) if path.is_file() else None

    def _log_missing_predictions(self, hp_item: HPItem):
        logging.warning(f"{self.__class__.__name__}: predictions file not found for split {hp_item.split_index}, "
                        f"encoder {hp_item.hp_setting.encoder_name}, ml_method {hp_item.hp_setting.ml_method_name}")

    def _process_dataset(self, dataset_info: dict, label: Label, hp_item: HPItem):
        predictions = dataset_info['predictions']
        metadata = dataset_info['metadata']
        
        # Calculate overall performance
        overall_performance = self._calculate_performance(predictions, label)
        
        # Calculate per-label performances and counts
        alt_label_performances = {}
        alt_label_counts = {}
        total_count = len(predictions)
        
        for alt_label_value in self.alternative_label_values:
            perf = self._get_performance_for_label_value(predictions, metadata, label, alt_label_value)
            count = self._get_count_for_label_value(metadata, alt_label_value)
            
            alt_label_performances[f"performance_{alt_label_value}"] = perf
            alt_label_counts[f"count_{alt_label_value}"] = count
        
        return {
            "setting": hp_item.hp_setting.get_key(),
            "performance": overall_performance,
            "example_count": total_count,
            **alt_label_performances,
            **alt_label_counts
        }

    def _get_performance_for_label_value(self, predictions, metadata, label, alt_label_value):
        mask = metadata[self.alternative_label] == alt_label_value
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            true = predictions[f"{label.name}_true_class"].iloc[indices]
            predicted = predictions[f"{label.name}_predicted_class"].iloc[indices]
            proba = self._get_probabilities(predictions, label)
            if proba is not None:
                proba = proba.iloc[indices]
            
            return self._compute_metric(true, predicted, proba, label)
        return float('nan')

    def _get_count_for_label_value(self, metadata, alt_label_value):
        return len(metadata[metadata[self.alternative_label] == alt_label_value])

    def _get_probabilities(self, predictions, label):
        proba_col = f"{label.name}_{label.positive_class}_proba"
        return predictions[proba_col] if proba_col in predictions.columns else None

    def _calculate_performance(self, predictions, label):
        true = predictions[f"{label.name}_true_class"]
        predicted = predictions[f"{label.name}_predicted_class"]
        proba = self._get_probabilities(predictions, label)
        
        return self._compute_metric(true, predicted, proba, label)

    def _compute_metric(self, true, predicted, proba, label):
        return MetricUtil.score_for_metric(
            metric=ClassificationMetric[self.metric.upper()],
            true_y=true,
            predicted_y=predicted,
            predicted_proba_y=proba,
            classes=label.values
        )

    def _write_split_results(self, data: Dict[str, pd.DataFrame], name_suffix: str, label_name: str):
        outputs = []
        datasets = []
        if self.plot_on_train:
            datasets.append('train')
        if self.plot_on_test:
            datasets.append('test')

        for dataset_desc in datasets:
            outputs.append(self._write_performance_tables(data[dataset_desc], dataset_desc, name_suffix + "_" + dataset_desc, label_name))
            outputs.append(self._create_performance_plot(data[dataset_desc], dataset_desc, name_suffix + "_" + dataset_desc, label_name))

        outputs = [x for item in outputs for x in (item if isinstance(item, list) else [item])]

        return outputs

    def _write_performance_tables(self, data: pd.DataFrame, dataset_desc: str, name_suffix: str, label_name: str):
        table_path = self.result_path / f"{name_suffix}_performance.csv"
        data.to_csv(table_path, index=False)
        return ReportOutput(table_path, self._get_desc_from_name_suffix(name_suffix, label_name, dataset_desc))

    def _get_desc_from_name_suffix(self, name_suffix: str, label_name: str, dataset_desc: str):
        return (f"Performance on {label_name} split by {self.alternative_label} ({dataset_desc} set "
                f"in {'-'.join([el for el in name_suffix.replace(label_name, '').replace(dataset_desc, '').split('_') if el != ''])})")

    def _reorder_columns(self, df):
        # Base columns
        cols = ["setting", "performance", "example_count"]
        
        # Add performance columns for each alternative label value
        perf_cols = [f"performance_{val}" for val in self.alternative_label_values]
        
        # Add count columns for each alternative label value
        count_cols = [f"count_{val}" for val in self.alternative_label_values]
        
        return df[cols + perf_cols + count_cols]

    def _create_performance_plot(self, data: pd.DataFrame, dataset_desc: str, name_suffix, label_name: str):
        fig = self._create_figure(data, label_name)
        plot_path = self.result_path / f"{name_suffix}_performance_plot.html"
        fig.write_html(str(plot_path))
        return ReportOutput(plot_path,
                            self._get_desc_from_name_suffix(name_suffix, label_name, dataset_desc))

    def _create_figure(self, data: pd.DataFrame, label_name: str):
        fig = go.Figure()
        groups_for_perf_eval = ['all'] + self.alternative_label_values
        repetitions = data.run_id.unique().shape[0]
        groups_for_perf_eval = np.repeat(groups_for_perf_eval, repetitions)

        for setting in data.setting.unique().tolist():
            setting_data = data[data.setting == setting]
            y = np.concatenate([setting_data.performance] + [setting_data[f'performance_{alt_lbl_value}'] for alt_lbl_value in self.alternative_label_values]).round(3)
            if repetitions > 1:
                fig.add_trace(go.Box(
                    name=setting,
                    x=groups_for_perf_eval,
                    y=y,
                    boxpoints='all',
                    jitter=0.3
                ))
            else:
                fig.add_trace(go.Bar(
                    name=setting,
                    x=groups_for_perf_eval,
                    y=y,
                    hovertemplate=setting + "<br>" + self.alternative_label + ": %{x}<br>" + self.metric + ": %{y}<extra></extra>"
                ))
        
        fig.update_layout(**self._get_layout_settings({'boxmode': 'group'} if repetitions > 1 else {'barmode': 'group'}))
        return fig

    def _get_layout_settings(self, kwargs):
        return {
            **kwargs,
            "title": f"Performance by {self.alternative_label}",
            "xaxis_title": self.alternative_label,
            "yaxis_title": f"{self.metric.replace('_', ' ').title()}",
            "template": "plotly_white",
            "showlegend": True
        }

    def check_prerequisites(self):
        if self.state is None:
            logging.warning(
                f"{self.__class__.__name__} requires state to be set. {self.__class__.__name__} report will not be created.")
            return False
        if not self.compute_for_selection and not self.compute_for_assessment:
            logging.warning(
                f"{self.__class__.__name__} requires either compute_for_selection or compute_for_assessment to be True. "
                f"{self.__class__.__name__} report will not be created.")
            return False
        return True
