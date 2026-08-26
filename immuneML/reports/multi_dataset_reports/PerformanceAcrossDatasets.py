import logging
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.multi_dataset_reports.MultiDatasetReport import MultiDatasetReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class PerformanceAcrossDatasets(MultiDatasetReport):
    """
    PerformanceAcrossDatasets report shows, for each hyperparameter setting (a combination of encoding and ML method,
    as specified under 'settings' of the TrainMLModel instruction used by MultiDatasetBenchmarkTool), how its
    performance varies across the different datasets that were benchmarked. This is useful, for example, to explore
    how model performance changes with dataset size, class balance, or any other property that differs between the
    datasets given to MultiDatasetBenchmarkTool.

    This report can be used only with MultiDatasetBenchmarkTool. Datasets are shown on the x-axis in the same order
    as they were listed under 'datasets' in the MultiDatasetBenchmarkTool specification, so that, for example, a
    trend across increasing dataset size or class balance can be examined by listing the datasets accordingly.

    One figure is created per performance metric, showing all hyperparameter settings together as separate lines (one
    color per setting); a single hyperparameter setting can be isolated by double-clicking its entry in the plot
    legend. Each marker shows the mean of the metric across the assessment (outer loop) cross-validation splits for
    that dataset, with error bars showing the spread across those splits.

    **Specification arguments:**

    - metrics (list): a list of metric names to show (e.g., [auc, accuracy]). For a given hyperparameter setting, if
      a metric was already computed as a part of the TrainMLModel instruction (that is, it is the optimization
      metric, or it is listed under 'metrics' there), the already computed value is reused; otherwise, it is computed
      from the stored test predictions. If this argument is not set, only the optimization metric is shown (which is
      always available).

    - error_bar (str): how to compute the error bars shown for each point, based on the values across the assessment
      splits; one of `standard_deviation` or `standard_error_of_mean`. Default is `standard_deviation`.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_performance_report:
                    PerformanceAcrossDatasets:
                        metrics: [auc, balanced_accuracy]
                        error_bar: standard_deviation

    """

    ERROR_BAR_OPTIONS = ["standard_deviation", "standard_error_of_mean"]

    @classmethod
    def build_object(cls, **kwargs):
        location = "PerformanceAcrossDatasets"

        metric_names = kwargs.get('metrics', None)
        if metric_names is not None:
            ParameterValidator.assert_type_and_value(metric_names, list, location, 'metrics')

        error_bar = kwargs.get('error_bar', 'standard_deviation')
        ParameterValidator.assert_in_valid_list(error_bar, PerformanceAcrossDatasets.ERROR_BAR_OPTIONS, location, 'error_bar')

        return PerformanceAcrossDatasets(metrics=metric_names, error_bar=error_bar, name=kwargs.get('name', None))

    def __init__(self, instruction_states: List[TrainMLModelState] = None, metrics: List[str] = None,
                 error_bar: str = "standard_deviation", name: str = None, result_path: Path = None,
                 number_of_processes: int = 1):
        super().__init__(instruction_states=instruction_states, name=name, result_path=result_path,
                         number_of_processes=number_of_processes)
        self.metric_names = metrics
        self.error_bar = error_bar
        self.label: Label = None
        self.metrics: List[ClassificationMetric] = None
        self.hp_settings: dict = None

    def _generate(self) -> ReportResult:
        self.result_path = PathBuilder.build(self.result_path / self.name)

        self._extract_label()
        self._resolve_metrics()
        self._resolve_hp_settings()

        long_df = self._collect_performance()
        summary_df = self._summarize(long_df)

        output_tables = self._write_tables(long_df, summary_df)
        output_figures = self._make_figures(summary_df)

        return ReportResult(name=self.name,
                            info="Shows how the performance of each hyperparameter setting varies across the "
                                 "datasets used with MultiDatasetBenchmarkTool.",
                            output_figures=output_figures, output_tables=output_tables)

    def _extract_label(self):
        all_labels = []
        for state in self.instruction_states:
            all_labels += state.label_configuration.get_label_objects()

        label_names = set(label.name for label in all_labels)
        assert len(label_names) == 1, \
            f"{PerformanceAcrossDatasets.__name__}: multiple labels were specified across the datasets ({label_names}), " \
            f"but this report accepts only one label."

        self.label = all_labels[0]

    def _resolve_metrics(self):
        if self.metric_names:
            self.metrics = [ClassificationMetric.get_metric(metric_name) for metric_name in self.metric_names]
        else:
            optimization_metrics = set(state.optimization_metric for state in self.instruction_states)
            assert len(optimization_metrics) == 1, \
                f"{PerformanceAcrossDatasets.__name__}: the datasets were optimized for different metrics " \
                f"({sorted(m.name for m in optimization_metrics)}); please set the 'metrics' argument explicitly to " \
                f"choose which metric(s) to show."
            self.metrics = list(optimization_metrics)

    def _resolve_hp_settings(self):
        hp_setting_keys_per_state = [frozenset(hp_setting.get_key() for hp_setting in state.hp_settings)
                                     for state in self.instruction_states]
        assert len(set(hp_setting_keys_per_state)) == 1, \
            f"{PerformanceAcrossDatasets.__name__}: the datasets used with MultiDatasetBenchmarkTool do not all use " \
            f"the same hyperparameter settings, cannot compare performance across datasets: {hp_setting_keys_per_state}."

        self.hp_settings = {hp_setting.get_key(): hp_setting for hp_setting in self.instruction_states[0].hp_settings}

    def _collect_performance(self) -> pd.DataFrame:
        rows = []
        for state in self.instruction_states:
            dataset_name = state.dataset.name
            for assessment_state in state.assessment_states:
                label_state = assessment_state.label_states[self.label.name]
                for hp_setting_key in self.hp_settings:
                    hp_item = label_state.assessment_items[hp_setting_key]
                    for metric in self.metrics:
                        value = self._get_metric_value(hp_item, metric)
                        rows.append({"dataset": dataset_name, "hp_setting": hp_setting_key,
                                    "metric": metric.name.lower(), "split_index": assessment_state.split_index,
                                    "value": value})

        return pd.DataFrame(rows)

    def _get_metric_value(self, hp_item: HPItem, metric: ClassificationMetric) -> float:
        metric_key = metric.name.lower()
        if hp_item.performance is not None and metric_key in hp_item.performance:
            return hp_item.performance[metric_key]
        return self._compute_metric_from_predictions(hp_item, metric)

    def _compute_metric_from_predictions(self, hp_item: HPItem, metric: ClassificationMetric) -> float:
        if hp_item.test_predictions_path is None or not Path(hp_item.test_predictions_path).is_file():
            setting_desc = hp_item.hp_setting.get_key() if hp_item.hp_setting is not None else "?"
            logging.warning(f"{PerformanceAcrossDatasets.__name__}: could not find test predictions for hyperparameter "
                            f"setting {setting_desc}, skipping metric {metric.name}.")
            return float('nan')

        predictions = pd.read_csv(hp_item.test_predictions_path)
        true_y = predictions[f"{self.label.name}_true_class"]
        predicted_y = predictions[f"{self.label.name}_predicted_class"]
        proba_col = f"{self.label.name}_{self.label.positive_class}_proba"
        predicted_proba_y = predictions[proba_col] if proba_col in predictions.columns else None

        return MetricUtil.score_for_metric(metric=metric, predicted_y=predicted_y, predicted_proba_y=predicted_proba_y,
                                           true_y=true_y, classes=self.label.values, pos_class=self.label.positive_class)

    def _summarize(self, long_df: pd.DataFrame) -> pd.DataFrame:
        grouped = long_df.groupby(["dataset", "hp_setting", "metric"], sort=False)["value"]
        summary = grouped.agg(mean="mean", std="std", n="count").reset_index()
        summary["sem"] = summary["std"] / summary["n"].pow(0.5)
        summary["error"] = (summary["std"] if self.error_bar == "standard_deviation" else summary["sem"]).fillna(0)
        return summary

    def _write_tables(self, long_df: pd.DataFrame, summary_df: pd.DataFrame) -> List[ReportOutput]:
        long_path = self.result_path / "performance_across_datasets_raw.csv"
        long_df.to_csv(long_path, index=False)

        summary_path = self.result_path / "performance_across_datasets_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        return [ReportOutput(long_path, "performance across datasets (per assessment split, csv)"),
                ReportOutput(summary_path, "performance across datasets (summary, csv)")]

    def _make_figures(self, summary_df: pd.DataFrame) -> List[ReportOutput]:
        outputs = []
        dataset_order = [state.dataset.name for state in self.instruction_states]
        hp_setting_keys = sorted(self.hp_settings.keys())
        colors = px.colors.qualitative.Vivid
        color_map = {key: colors[index % len(colors)] for index, key in enumerate(hp_setting_keys)}

        for metric in self.metrics:
            metric_name = metric.name.lower()
            metric_df = summary_df[summary_df["metric"] == metric_name]

            figure = go.Figure()
            for hp_setting_key in hp_setting_keys:
                setting_df = metric_df[metric_df["hp_setting"] == hp_setting_key].set_index("dataset").reindex(dataset_order).reset_index()
                figure.add_trace(go.Scatter(x=setting_df["dataset"], y=setting_df["mean"], mode="lines+markers",
                                            name=hp_setting_key, line=dict(color=color_map[hp_setting_key]),
                                            error_y=dict(type="data", array=setting_df["error"], visible=True)))

            figure.update_layout(template="plotly_white", xaxis_title="dataset",
                                 yaxis_title=metric_name.replace("_", " "),
                                 title=f"{metric_name.replace('_', ' ').title()} across datasets")
            figure.update_xaxes(categoryorder="array", categoryarray=dataset_order)

            figure_path = self.result_path / f"performance_across_datasets_{metric_name}.html"
            figure_path = PlotlyUtil.write_image_to_file(figure, figure_path)
            outputs.append(ReportOutput(figure_path, f"{metric_name.replace('_', ' ')} across datasets"))

        return outputs

    def check_prerequisites(self):
        if not self.instruction_states or len(self.instruction_states) < 2:
            logging.warning(f"{PerformanceAcrossDatasets.__name__} requires at least 2 datasets (instruction states) "
                            f"to compare performance across, report will not be created.")
            return False
        return True