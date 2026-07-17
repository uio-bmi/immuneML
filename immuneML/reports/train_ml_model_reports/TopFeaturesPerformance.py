import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_metrics.MetricUtil import MetricUtil
from immuneML.ml_methods.util.Util import Util
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.train_ml_model_reports.TrainMLModelReport import TrainMLModelReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class TopFeaturesPerformance(TrainMLModelReport):
    """
    Report for the :ref:`TrainMLModel` instruction: produces a performance saturation curve showing how
    model performance changes as progressively more features are included, ranked by coefficient magnitude
    (for linear models) or feature importance (for tree-based models).

    For each outer cross-validation split, the report takes an already-trained model and evaluates it
    on the test set while zeroing out all but the top-N input features. This is repeated for each value
    of N in ``top_n_values``. The rightmost point is always the full model (all features) and serves as
    the reference. The resulting curve reveals how many features are actually needed to achieve near-optimal
    performance — the "knee" of the saturation curve.

    In addition, for each N the report counts how many CV splits include each feature in the top-N set.
    This shows which features are consistently ranked highly across splits (appearing in many splits) and
    which are borderline (appearing in only a few).

    Compatible models: LogisticRegression, SVM, SVC, LogRegressionCustomPenalty (ranked by ``|coef_|``),
    and RandomForestClassifier, GradientBoosting, XGBClassifier (ranked by ``feature_importances_``).

    **Two comparison modes** (controlled by ``use_optimal_only``):

    - ``use_optimal_only: true`` *(default)*: evaluates only the optimal HP setting per split.
      The plot shows one line per CV split plus a mean ± std band — useful for assessing consistency
      of the saturation pattern across dataset splits.

    - ``use_optimal_only: false``: evaluates every HP setting (encoding + ML method combination) in
      the assessment loop across all splits. The plot shows one mean ± std curve per HP setting —
      useful for comparing whether different model configurations require the same number of features.


    **Specification arguments:**

    - top_n_values (list): integers specifying how many top features to include at each evaluation point,
      e.g. ``[5, 10, 25, 50, 100]``. Values exceeding the total feature count are ignored; the full model
      is always included as the final point. Default: ``[10, 25, 50, 100]``.

    - use_optimal_only (bool): if ``true``, only the selected optimal HP setting is evaluated per split.
      If ``false``, all HP settings are evaluated and grouped in the plot. Default: ``true``.

    - metric (str): name of the :py:obj:`~immuneML.ml_metrics.ClassificationMetric.ClassificationMetric`
      to use. By default, the optimization metric from the :ref:`TrainMLModel` instruction is used.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_saturation_report:
                    TopFeaturesPerformance:
                        top_n_values: [5, 10, 25, 50, 100]
                        use_optimal_only: true
                        metric: BALANCED_ACCURACY  # optional

    """

    REPORT_INFO = (
        "Examines how model performance depends on the number of top-ranked features (performance saturation), "
        "and how often each feature appears in the top-N set across cross-validation splits. "
        "Supports linear models (ranked by |coefficient|) and tree-based models (ranked by feature importance)."
    )

    @classmethod
    def build_object(cls, **kwargs):
        location = "TopFeaturesPerformance"

        top_n_values = kwargs.get("top_n_values", None)
        ParameterValidator.assert_type_and_value(top_n_values, list, location, "top_n_values")
        ParameterValidator.assert_all_type_and_value(top_n_values, int, location, "top_n_values", min_inclusive=1)

        use_optimal_only = kwargs.get("use_optimal_only", None)
        ParameterValidator.assert_type_and_value(use_optimal_only, bool, location, "use_optimal_only")

        metric = kwargs.get("metric", None)
        if metric is not None:
            ParameterValidator.assert_type_and_value(metric, str, location, "metric")

        return TopFeaturesPerformance(
            top_n_values=top_n_values,
            use_optimal_only=use_optimal_only,
            metric=metric,
            name=kwargs.get("name"),
        )

    def __init__(self, top_n_values: List[int] = None, use_optimal_only: bool = True,
                 metric: Optional[str] = None, name: str = None,
                 state: TrainMLModelState = None, label: Label = None,
                 result_path: Path = None, number_of_processes: int = 1):
        super().__init__(name=name, state=state, label=label, result_path=result_path,
                         number_of_processes=number_of_processes)
        self.top_n_values = sorted(top_n_values)
        self.use_optimal_only = use_optimal_only
        self.metric = metric

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
        metric = self._resolve_metric()
        output_tables, output_figures = [], []

        for label in self.state.label_configuration.get_label_objects():
            tables, perf_figures = self._generate_performance_outputs_for_label(label, metric)
            output_tables.extend(tables)
            output_figures.extend(perf_figures)
            stab_tables, stab_figures = self._generate_stability_outputs_for_label(label)
            output_tables.extend(stab_tables)
            output_figures.extend(stab_figures)

        return ReportResult(name=self.name, info=self.REPORT_INFO,
                            output_tables=output_tables, output_figures=output_figures)

    def _generate_performance_outputs_for_label(
            self, label: Label, metric: ClassificationMetric
    ) -> Tuple[list, list]:
        rows = self._collect_rows_for_label(label, metric)
        if not rows:
            logging.warning(
                f"{self.__class__.__name__}: no data collected for label '{label.name}'. "
                f"Check that at least one model exposes coef_ or feature_importances_."
            )
            return [], []
        df = pd.DataFrame(rows)
        csv_path = self.result_path / f"top_features_performance_{label.name}.csv"
        df.to_csv(csv_path, index=False)
        tables = [ReportOutput(csv_path,
                               name=f"Performance data ({metric.name.lower()}) vs. number of top features — label: {label.name}")]
        figures = [fig for fig in [self._plot_performance_curve(df, label, metric)] if fig is not None]
        return tables, figures

    def _generate_stability_outputs_for_label(self, label: Label) -> Tuple[list, list]:
        feature_sets = self._collect_feature_sets_for_label(label)
        if not feature_sets:
            return [], []
        n_splits = self.state.assessment.split_count
        tables = self._export_frequency_table(feature_sets, label, n_splits)
        figures = [fig for fig in [self._plot_selection_frequency(feature_sets, label)]
                   if fig is not None]
        return tables, figures

    def _export_frequency_table(self, feature_sets: dict, label: Label, n_splits: int) -> list:
        freq_rows = []
        for hp_key, hp_data in feature_sets.items():
            for n, sets in hp_data['sets'].items():
                counts = {}
                for feat_set in sets:
                    for feat in feat_set:
                        counts[feat] = counts.get(feat, 0) + 1
                for feat, count in counts.items():
                    freq_rows.append({'label': label.name, 'hp_setting': hp_key,
                                      'n_features': n, 'feature': feat,
                                      'selection_count': count, 'n_splits': n_splits})
        if not freq_rows:
            return []
        path = self.result_path / f"feature_selection_frequency_{label.name}.csv"
        (pd.DataFrame(freq_rows)
         .sort_values(['hp_setting', 'n_features', 'selection_count'], ascending=[True, True, False])
         .to_csv(path, index=False))
        return [ReportOutput(path, name=f"Feature selection frequency — label: {label.name}")]

    def _resolve_metric(self) -> ClassificationMetric:
        if self.metric is not None:
            return ClassificationMetric.get_metric(self.metric)
        return self.state.optimization_metric

    def _collect_rows_for_label(self, label: Label, metric: ClassificationMetric) -> list:
        rows = []
        for split_idx, assessment_state in enumerate(self.state.assessment_states):
            label_state = assessment_state.label_states.get(label.name)
            if label_state is None:
                continue
            for _, hp_item in self._get_hp_items_for_label_state(label_state):
                rows.extend(self._evaluate_hp_item(hp_item, split_idx, label, metric))
        return rows

    def _evaluate_hp_item(self, hp_item: HPItem, split_idx: int,
                          label: Label, metric: ClassificationMetric) -> list:
        encoded_data, importances = self._get_encoded_data_and_importances(hp_item)
        if encoded_data is None:
            logging.warning(f"{self.__class__.__name__}: test dataset has no encoded data "
                            f"for split {split_idx}, setting '{hp_item.hp_setting}'.")
            return []
        if importances is None:
            logging.warning(f"{self.__class__.__name__}: model '{type(hp_item.method).__name__}' "
                            f"does not expose coef_ or feature_importances_ — "
                            f"skipping split {split_idx}, setting '{hp_item.hp_setting}'.")
            return []
        X_test = encoded_data.get_examples_as_np_matrix()
        y_true = np.array(encoded_data.labels[label.name])
        n_features = X_test.shape[1]
        return self._score_top_n_subsets(hp_item.method, X_test, y_true, importances,
                                         label, metric, split_idx, str(hp_item.hp_setting), n_features)

    def _score_top_n_subsets(self, method, X_test: np.ndarray, y_true: np.ndarray,
                              importances: np.ndarray, label: Label, metric: ClassificationMetric,
                              split_idx: int, hp_key: str, n_features: int) -> list:
        rows = []
        sorted_idx = np.argsort(importances)
        for n in self._get_top_n_values(n_features):
            X_masked = np.zeros_like(X_test)
            X_masked[:, sorted_idx[-n:]] = X_test[:, sorted_idx[-n:]]
            score = self._compute_metric(method, X_masked, y_true, label, metric)
            if score is not None:
                rows.append({"split": split_idx + 1, "n_features": n, "performance": score,
                             "label": label.name, "hp_setting": hp_key,
                             "total_features": n_features, "is_full_model": n == n_features})
        return rows

    # ------------------------------------------------------------------ #
    # Feature stability — data collection                                 #
    # ------------------------------------------------------------------ #

    def _collect_feature_sets_for_label(self, label: Label) -> dict:
        """Returns {hp_key: {'sets': {n: [frozenset_per_split]}, 'd': int}}"""
        result = {}
        for assessment_state in self.state.assessment_states:
            label_state = assessment_state.label_states.get(label.name)
            if label_state is None:
                continue
            for hp_key, hp_item in self._get_hp_items_for_label_state(label_state):
                item_data = self._build_top_n_feature_sets(hp_item)
                if item_data is None:
                    continue
                if hp_key not in result:
                    result[hp_key] = {'sets': {}, 'd': item_data['d']}
                for n, feat_set in item_data['sets'].items():
                    result[hp_key]['sets'].setdefault(n, []).append(feat_set)
        return result

    def _build_top_n_feature_sets(self, hp_item: HPItem) -> Optional[dict]:
        encoded_data, importances = self._get_encoded_data_and_importances(hp_item)
        if encoded_data is None or importances is None:
            return None
        n_features = len(importances)
        feature_names = (list(encoded_data.feature_names) if encoded_data.feature_names is not None
                         else [str(i) for i in range(n_features)])
        sorted_idx = np.argsort(importances)
        n_values = [n for n in self.top_n_values if n < n_features]
        if not n_values:
            return None
        sets = {n: frozenset(feature_names[i] for i in sorted_idx[-n:]) for n in n_values}
        return {'sets': sets, 'd': n_features}

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    def _get_hp_items_for_label_state(self, label_state) -> list:
        if self.use_optimal_only:
            try:
                return [("optimal", label_state.optimal_assessment_item)]
            except Exception:
                return []
        return [(str(item.hp_setting), item) for item in label_state.assessment_items.values()]

    def _get_encoded_data_and_importances(self, hp_item: HPItem) -> tuple:
        encoded_data = hp_item.test_dataset.encoded_data if hp_item.test_dataset is not None else None
        if encoded_data is None:
            return None, None
        return encoded_data, self._get_feature_importances(hp_item.method)

    def _get_top_n_values(self, n_features: int) -> list:
        return sorted(set([n for n in self.top_n_values if n < n_features] + [n_features]))

    def _get_feature_importances(self, method) -> Optional[np.ndarray]:
        try:
            params = method.get_params(for_refitting=False)
            if "coefficients" in params:
                return np.abs(np.array(params["coefficients"], dtype=float))
            if "feature_importances" in params:
                return np.array(params["feature_importances"], dtype=float)
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not extract feature importances: {e}")
        return None

    def _compute_metric(self, method, X_masked: np.ndarray, y_true: np.ndarray,
                        label: Label, metric: ClassificationMetric) -> Optional[float]:
        try:
            raw_pred = method.model.predict(X_masked)
            y_pred = Util.map_to_old_class_values(np.array(raw_pred), method.class_mapping)
            y_proba = self._get_positive_class_proba(method, X_masked, label)
            score = MetricUtil.score_for_metric(metric=metric, predicted_y=y_pred,
                                                predicted_proba_y=y_proba, true_y=y_true,
                                                classes=label.values, pos_class=label.positive_class)
            return float(score) if isinstance(score, (int, float)) else None
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not compute metric: {e}")
            return None

    def _get_positive_class_proba(self, method, X_masked: np.ndarray,
                                  label: Label) -> Optional[np.ndarray]:
        if not method.can_predict_proba() or label.positive_class is None:
            return None
        original_to_internal = {v: k for k, v in method.class_mapping.items()}
        pos_internal = original_to_internal.get(label.positive_class)
        if pos_internal is None:
            return None
        classes_list = list(method.model.classes_)
        if pos_internal not in classes_list:
            return None
        return method.model.predict_proba(X_masked)[:, classes_list.index(pos_internal)]

    # ------------------------------------------------------------------ #
    # Performance saturation — plotting                                   #
    # ------------------------------------------------------------------ #

    def _plot_performance_curve(self, df: pd.DataFrame, label: Label,
                                metric: ClassificationMetric) -> Optional[ReportOutput]:
        try:
            figure = (self._plot_by_split(df, label, metric) if self.use_optimal_only
                      else self._plot_by_hp_setting(df, label, metric))
            file_path = PlotlyUtil.write_image_to_file(
                figure, self.result_path / f"top_features_performance_{label.name}.html")
            return ReportOutput(path=file_path,
                                name=f"Performance saturation curve ({metric.name.lower()}): how "
                                     f"{metric.name.lower()} changes as more top-ranked features are "
                                     f"included — label: {label.name}")
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not generate performance plot: {e}")
            return None

    def _plot_by_split(self, df: pd.DataFrame, label: Label,
                       metric: ClassificationMetric) -> go.Figure:
        metric_name = metric.name.replace("_", " ").title()
        figure = go.Figure()
        palette = px.colors.qualitative.Vivid
        splits = sorted(df["split"].unique())
        common_x = sorted(df["n_features"].unique())

        for i, split in enumerate(splits):
            split_df = df[df["split"] == split].sort_values("n_features")
            figure.add_trace(go.Scatter(
                x=split_df["n_features"].tolist(), y=split_df["performance"].tolist(),
                mode="lines+markers", name=f"Split {split}", opacity=0.5,
                line=dict(color=palette[i % len(palette)], width=1), marker=dict(size=5),
                hovertemplate=f"Split {split}<br>Features: %{{x}}<br>{metric_name}: %{{y:.4f}}<extra></extra>"))

        if len(splits) > 1:
            interp_y = self._interpolate_per_split(df, splits, common_x)
            mean_y, std_y = np.mean(interp_y, axis=0), np.std(interp_y, axis=0)
            self._add_mean_std_band(figure, common_x, mean_y, std_y, "Mean", "#1a1aff", metric_name)

        self._apply_layout(figure, metric_name, label.name)
        return figure

    def _plot_by_hp_setting(self, df: pd.DataFrame, label: Label,
                            metric: ClassificationMetric) -> go.Figure:
        metric_name = metric.name.replace("_", " ").title()
        figure = go.Figure()
        palette = px.colors.qualitative.Vivid

        for i, hp_key in enumerate(sorted(df["hp_setting"].unique())):
            hp_df = df[df["hp_setting"] == hp_key]
            common_x = sorted(hp_df["n_features"].unique())
            splits = sorted(hp_df["split"].unique())
            interp_y = self._interpolate_per_split(hp_df, splits, common_x)
            mean_y = np.mean(interp_y, axis=0)
            std_y = np.std(interp_y, axis=0) if len(splits) > 1 else np.zeros_like(mean_y)
            self._add_mean_std_band(figure, common_x, mean_y, std_y, hp_key,
                                    palette[i % len(palette)], metric_name)

        self._apply_layout(figure, metric_name, label.name)
        return figure

    def _interpolate_per_split(self, df: pd.DataFrame, splits: list,
                               common_x: list) -> list:
        interp_y = []
        for split in splits:
            split_df = df[df["split"] == split].sort_values("n_features")
            interp_y.append(np.interp(common_x, split_df["n_features"].tolist(),
                                      split_df["performance"].tolist()))
        return interp_y

    def _add_mean_std_band(self, figure: go.Figure, x: list, mean_y: np.ndarray,
                           std_y: np.ndarray, name: str, color: str, metric_name: str):
        r, g, b = self._hex_to_rgb(color)
        figure.add_trace(go.Scatter(
            x=x + x[::-1], y=list(mean_y + std_y) + list((mean_y - std_y)[::-1]),
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(255,255,255,0)"), showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(
            x=x, y=mean_y, mode="lines+markers", name=name,
            line=dict(color=color, width=3), marker=dict(size=7),
            hovertemplate=name + "<br>Features: %{x}<br>" + metric_name + ": %{y:.4f}<extra></extra>"))

    @staticmethod
    def _apply_layout(figure: go.Figure, metric_name: str, label_name: str):
        figure.update_layout(
            template="plotly_white", xaxis_title="Number of top features used",
            yaxis_title=metric_name, title=f"Top-features performance — label: {label_name}",
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99), font_size=14)

    @staticmethod
    def _hex_to_rgb(color: str) -> tuple:
        try:
            color = color.lstrip("#")
            if len(color) == 6:
                return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        except Exception:
            pass
        return 128, 128, 128

    # ------------------------------------------------------------------ #
    # Feature stability — plotting                                        #
    # ------------------------------------------------------------------ #

    def _plot_selection_frequency(self, feature_sets: dict,
                                  label: Label) -> Optional[ReportOutput]:
        try:
            all_n_values = sorted({n for hp_data in feature_sets.values() for n in hp_data['sets']})
            if not all_n_values:
                return None
            n_splits = self.state.assessment.split_count
            fig = make_subplots(rows=len(all_n_values), cols=1,
                                subplot_titles=[f"Top {n} features"
                                                for n in all_n_values])
            for row_idx, n in enumerate(all_n_values, start=1):
                counts = self._count_feature_selections(feature_sets, n)
                if counts:
                    self._add_frequency_bar(fig, counts, n_splits, row_idx, n)
            fig.update_layout(template="plotly_white",
                              title=f"Feature selection frequency — label: {label.name}",
                              height=350 * len(all_n_values), font_size=12)
            file_path = PlotlyUtil.write_image_to_file(
                fig, self.result_path / f"feature_selection_frequency_{label.name}.html")
            return ReportOutput(path=file_path,
                                name=f"Feature selection frequency: how many CV splits include each "
                                     f"feature in the top N, sorted by count (tick labels shown for "
                                     f"≤30 features; names available on hover) — label: {label.name}")
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not generate selection frequency plot: {e}")
            return None

    def _count_feature_selections(self, feature_sets: dict, n: int) -> dict:
        counts = {}
        for hp_data in feature_sets.values():
            for feat_set in hp_data['sets'].get(n, []):
                for feat in feat_set:
                    counts[feat] = counts.get(feat, 0) + 1
        return counts

    def _add_frequency_bar(self, fig: go.Figure, counts: dict,
                           n_splits: int, row_idx: int, n: int):
        sorted_feats = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        features = [f for f, _ in sorted_feats]
        freqs = [c for _, c in sorted_feats]
        fig.add_trace(
            go.Bar(x=list(range(len(features))), y=freqs, marker_color='steelblue', showlegend=False,
                   customdata=features,
                   hovertemplate='%{customdata}<br>Selected in %{y}/' + str(n_splits) + ' runs<extra></extra>'),
            row=row_idx, col=1)
        if len(features) <= 30:
            fig.update_xaxes(tickvals=list(range(len(features))), ticktext=features,
                             tickangle=45, col=1, row=row_idx)
        else:
            fig.update_xaxes(showticklabels=False, col=1, row=row_idx)
        fig.update_yaxes(title_text=f"Selection count (out of {n_splits})",
                         range=[0, n_splits], col=1, row=row_idx)

