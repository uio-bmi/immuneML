import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder

ZERO_THRESHOLD = 1e-8
MEAN_STD_COLOR = "#1a1aff"


class MLParameterPerformance(TrainMLModelReport):
    """
    Report for the :ref:`TrainMLModel` instruction: shows how test performance varies across a set of
    ``ml_methods`` HP settings that differ only in one named hyperparameter (e.g. a regularization path
    over ``C`` or ``alpha``, a tree depth, an optimizer choice — any hyperparameter name works).
    Performance is aggregated across the outer cross-validation splits: one line (or box) per split plus
    a mean ± SD summary. One figure and table are produced per metric in ``metrics``.

    For HP settings whose fitted model exposes ``coefficients`` or ``feature_importances`` via
    ``get_params()`` (e.g. linear or tree-based models), the report also counts non-zero coefficients per
    setting and produces a second plot of performance vs. number of non-zero features.

    (The report reads results from the models that TrainMLModel's nested cross-validation already fits
    and evaluates; it does not fit or evaluate anything itself.)

    **Specification arguments:**

    - hp_parameter_name (str): the name of the hyperparameter (as specified under the corresponding
      ``ml_methods`` entry in the YAML) whose value should be used to group and order HP settings, e.g.
      ``C``. Only HP settings whose ``ml_method`` was configured with this parameter are included.

    - metrics (list): names of :py:obj:`~immuneML.ml_metrics.ClassificationMetric.ClassificationMetric`
      to plot, one figure and table per metric. Each must be present in the :ref:`TrainMLModel`
      instruction's ``metrics`` list (or be the optimization metric). By default, only the optimization
      metric is used.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            ml_methods:
                log_reg_c_0.001:
                    LogisticRegression:
                        C: 0.001
                log_reg_c_0.01:
                    LogisticRegression:
                        C: 0.01
                # ... one entry per C value along the path
            reports:
                my_regularization_path_report:
                    MLParameterPerformance:
                        hp_parameter_name: C
                        metrics: [BALANCED_ACCURACY, AUC]  # optional

    """

    REPORT_INFO = (
        "Shows how test performance varies across a set of HP settings that differ only in one named "
        "hyperparameter (e.g. a regularization path), aggregated across the outer cross-validation splits "
        "of the TrainMLModel instruction. Also shows performance vs. number of non-zero features where the "
        "fitted models expose coefficients or feature importances."
    )

    @classmethod
    def build_object(cls, **kwargs):
        location = "MLParameterPerformance"

        hp_parameter_name = kwargs.get("hp_parameter_name", None)
        ParameterValidator.assert_type_and_value(hp_parameter_name, str, location, "hp_parameter_name")

        metrics = kwargs.get("metrics", None)
        if metrics is not None:
            ParameterValidator.assert_type_and_value(metrics, list, location, "metrics")
            ParameterValidator.assert_all_type_and_value(metrics, str, location, "metrics")

        return MLParameterPerformance(hp_parameter_name=hp_parameter_name, metrics=metrics, name=kwargs.get("name"))

    def __init__(self, hp_parameter_name: str = None, metrics: List[str] = None, name: str = None,
                 state: TrainMLModelState = None, label: Label = None, result_path: Path = None,
                 number_of_processes: int = 1):
        super().__init__(name=name, state=state, label=label, result_path=result_path,
                         number_of_processes=number_of_processes)
        self.hp_parameter_name = hp_parameter_name
        self.metrics = metrics

    def check_prerequisites(self) -> bool:
        if self.state is None:
            logging.warning(f"{self.__class__.__name__}: can only run as a TrainMLModel report (state is None).")
            return False
        if self.result_path is None:
            logging.warning(f"{self.__class__.__name__}: result_path is not set.")
            return False
        return True

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        output_tables, output_figures = [], []

        for label in self.state.label_configuration.get_label_objects():
            for metric in self._resolve_metrics():
                tables, figures = self._process_label_metric(label, metric)
                output_tables.extend(tables)
                output_figures.extend(figures)

        return ReportResult(name=self.name, info=self.REPORT_INFO,
                            output_tables=output_tables, output_figures=output_figures)

    def _resolve_metrics(self) -> List[ClassificationMetric]:
        if self.metrics:
            return [ClassificationMetric.get_metric(m) for m in self.metrics]
        return [self.state.optimization_metric]

    def _process_label_metric(self, label: Label, metric: ClassificationMetric) -> tuple:
        df = pd.DataFrame(self._collect_rows(label, metric))
        if df.empty:
            logging.warning(f"{self.__class__.__name__}: no HP settings with parameter "
                            f"'{self.hp_parameter_name}' and a computed '{metric.name.lower()}' score "
                            f"were found for label '{label.name}'.")
            return [], []

        csv_path = self.result_path / f"ml_parameter_performance_{label.name}_{metric.name.lower()}.csv"
        df.to_csv(csv_path, index=False)
        tables = [ReportOutput(csv_path,
                               name=f"Performance data ({metric.name.lower()}) vs. '{self.hp_parameter_name}' "
                                    f"— label: {label.name}")]

        figures = [fig for fig in [self._plot_vs_hp_value(df, label, metric),
                                   self._plot_vs_n_features(df, label, metric)] if fig is not None]
        return tables, figures

    # ------------------------------------------------------------------ #
    # Data collection                                                      #
    # ------------------------------------------------------------------ #

    def _collect_rows(self, label: Label, metric: ClassificationMetric) -> list:
        rows = []
        for split_idx, assessment_state in enumerate(self.state.assessment_states):
            label_state = assessment_state.label_states.get(label.name)
            if label_state is None:
                continue
            for hp_key, hp_item in label_state.assessment_items.items():
                row = self._build_row(hp_item, split_idx, label, metric, hp_key)
                if row is not None:
                    rows.append(row)
        return rows

    def _build_row(self, hp_item: HPItem, split_idx: int, label: Label,
                   metric: ClassificationMetric, hp_key: str) -> Optional[dict]:
        ml_params = hp_item.hp_setting.ml_params if hp_item.hp_setting is not None else None
        if not ml_params or self.hp_parameter_name not in ml_params:
            return None

        performance = hp_item.performance.get(metric.name.lower()) if hp_item.performance else None
        if performance is None:
            logging.warning(f"{self.__class__.__name__}: metric '{metric.name.lower()}' not found for split "
                            f"{split_idx}, setting '{hp_key}' — make sure it is included in TrainMLModel's "
                            f"'metrics' list. Skipping this HP setting.")
            return None

        return {"split": split_idx + 1, "hp_setting": hp_key, "hp_value": ml_params[self.hp_parameter_name],
                "n_features": self._count_nonzero_features(hp_item.method),
                "performance": float(performance), "label": label.name}

    @staticmethod
    def _count_nonzero_features(method) -> Optional[int]:
        try:
            params = method.get_params(for_refitting=False)
            if "coefficients" in params:
                values = params["coefficients"]
            elif "feature_importances" in params:
                values = params["feature_importances"]
            else:
                return None
            return int(np.sum(np.abs(np.array(values, dtype=float)) > ZERO_THRESHOLD))
        except Exception as e:
            logging.warning(f"MLParameterPerformance: could not extract feature importances: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Performance vs. hyperparameter value                                #
    # ------------------------------------------------------------------ #

    def _plot_vs_hp_value(self, df: pd.DataFrame, label: Label,
                          metric: ClassificationMetric) -> Optional[ReportOutput]:
        try:
            metric_name = self._metric_display_name(metric)
            figure = self._build_numeric_figure(df, metric_name) if self._is_numeric(df["hp_value"]) \
                else self._build_categorical_figure(df, metric_name)
            return self._finalize_figure(
                figure, title=f"Performance vs. {self.hp_parameter_name} — label: {label.name}",
                xaxis_title=self.hp_parameter_name, yaxis_title=metric_name,
                filename=f"ml_parameter_performance_{label.name}_{metric.name.lower()}.html",
                output_name=f"Performance ({metric.name.lower()}) vs. '{self.hp_parameter_name}' "
                           f"— label: {label.name}")
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not generate performance plot: {e}")
            return None

    @staticmethod
    def _is_numeric(values: pd.Series) -> bool:
        try:
            values.astype(float)
            return True
        except (TypeError, ValueError):
            return False

    def _build_numeric_figure(self, df: pd.DataFrame, metric_name: str) -> go.Figure:
        plot_df = df.copy()
        plot_df["hp_value"] = plot_df["hp_value"].astype(float)

        figure = go.Figure()
        for i, split in enumerate(sorted(plot_df["split"].unique())):
            self._add_split_trace(figure, plot_df[plot_df["split"] == split].sort_values("hp_value"),
                                  "hp_value", f"Split {split}", i, self.hp_parameter_name, metric_name)

        summary = plot_df.groupby("hp_value")["performance"].agg(["mean", "std"]).reset_index() \
            .sort_values("hp_value")
        self._add_mean_std_band(figure, summary["hp_value"].tolist(), summary["mean"].to_numpy(),
                                summary["std"].fillna(0.0).to_numpy(), "Mean ± SD",
                                metric_name, self.hp_parameter_name)

        values = plot_df["hp_value"]
        if (values > 0).all() and values.max() / max(values.min(), 1e-12) > 100:
            figure.update_xaxes(type="log")
        return figure

    def _build_categorical_figure(self, df: pd.DataFrame, metric_name: str) -> go.Figure:
        plot_df = df.copy()
        plot_df["hp_value"] = plot_df["hp_value"].astype(str)
        order = list(dict.fromkeys(plot_df.sort_values("split")["hp_value"]))
        return px.box(plot_df, x="hp_value", y="performance", points="all",
                     category_orders={"hp_value": order},
                     labels={"hp_value": self.hp_parameter_name, "performance": metric_name})

    # ------------------------------------------------------------------ #
    # Performance vs. number of non-zero features                         #
    # ------------------------------------------------------------------ #

    def _plot_vs_n_features(self, df: pd.DataFrame, label: Label,
                            metric: ClassificationMetric) -> Optional[ReportOutput]:
        feat_df = df[df["n_features"].notna()]
        if feat_df.empty:
            return None
        try:
            metric_name = self._metric_display_name(metric)
            figure = self._build_n_features_figure(feat_df, metric_name)
            return self._finalize_figure(
                figure, title=f"Performance vs. number of non-zero features — label: {label.name}",
                xaxis_title="Number of non-zero features", yaxis_title=metric_name,
                filename=f"ml_parameter_performance_{label.name}_{metric.name.lower()}_n_features.html",
                output_name=f"Performance ({metric.name.lower()}) vs. number of non-zero features "
                           f"— label: {label.name}")
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not generate n_features plot: {e}")
            return None

    def _build_n_features_figure(self, df: pd.DataFrame, metric_name: str) -> go.Figure:
        figure = go.Figure()
        per_split = [df[df["split"] == s].groupby("n_features", as_index=False)["performance"].mean()
                    .sort_values("n_features") for s in sorted(df["split"].unique())]
        for i, split_df in enumerate(per_split):
            self._add_split_trace(figure, split_df, "n_features", f"Split {i + 1}", i,
                                  "Number of non-zero features", metric_name)

        common_x = sorted(set(np.concatenate([sp["n_features"].to_numpy() for sp in per_split])))
        interp_y = [np.interp(common_x, sp["n_features"], sp["performance"]) for sp in per_split]
        self._add_mean_std_band(figure, common_x, np.mean(interp_y, axis=0), np.std(interp_y, axis=0),
                                "Mean ± SD", metric_name, "Number of non-zero features")
        return figure

    # ------------------------------------------------------------------ #
    # Shared plotting helpers                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _metric_display_name(metric: ClassificationMetric) -> str:
        return metric.name.replace("_", " ").title()

    def _finalize_figure(self, figure: go.Figure, title: str, xaxis_title: str, yaxis_title: str,
                         filename: str, output_name: str) -> ReportOutput:
        figure.update_layout(template="plotly_white", xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                             title=title, legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                             font_size=14)
        file_path = PlotlyUtil.write_image_to_file(figure, self.result_path / filename)
        return ReportOutput(path=file_path, name=output_name)

    @staticmethod
    def _add_split_trace(figure: go.Figure, split_df: pd.DataFrame, x_col: str, name: str,
                         color_idx: int, x_name: str, metric_name: str):
        palette = px.colors.qualitative.Vivid
        figure.add_trace(go.Scatter(
            x=split_df[x_col].tolist(), y=split_df["performance"].tolist(), mode="lines+markers",
            name=name, opacity=0.5, line=dict(color=palette[color_idx % len(palette)], width=1),
            marker=dict(size=5),
            hovertemplate=f"{name}<br>{x_name}: %{{x}}<br>{metric_name}: %{{y:.4f}}<extra></extra>"))

    @staticmethod
    def _add_mean_std_band(figure: go.Figure, x: list, mean_y: np.ndarray, std_y: np.ndarray,
                           name: str, metric_name: str, x_name: str):
        r, g, b = MLParameterPerformance._hex_to_rgb(MEAN_STD_COLOR)
        figure.add_trace(go.Scatter(
            x=x + x[::-1], y=list(mean_y + std_y) + list((mean_y - std_y)[::-1]),
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(255,255,255,0)"), showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(
            x=x, y=mean_y, mode="lines+markers", name=name, line=dict(color=MEAN_STD_COLOR, width=3),
            marker=dict(size=7),
            hovertemplate=f"{name}<br>{x_name}: " + "%{x}<br>" + f"{metric_name}: " + "%{y:.4f}<extra></extra>"))

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple:
        color = color.lstrip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)