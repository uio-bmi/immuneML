import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_methods.classifiers.SklearnMethod import SklearnMethod
from immuneML.ml_methods.classifiers.XGBClassifier import XGBClassifier
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder


class SHAPReport(MLReport):
    """
    A report that computes SHAP (SHapley Additive exPlanations) values for a trained sklearn-based classifier.

    SHAP values quantify how much each feature pushes a prediction above or below the expected model output.
    This report works with any classifier inheriting
    :py:obj:`~immuneML.ml_methods.classifiers.SklearnMethod.SklearnMethod`
    (e.g. :ref:`LogisticRegression`, :ref:`RandomForestClassifier`, :ref:`GradientBoosting`, :ref:`SVM`, :ref:`SVC`).

    The explainer is chosen automatically by :class:`shap.Explainer`: TreeExplainer for tree-based models,
    LinearExplainer for linear models, and KernelExplainer (with a background sample from the training set)
    for all others.

    Two plots are produced:

    - **Beeswarm** – one dot per test sample per feature; x-axis is the SHAP value. Conveys both direction
      and distribution of feature effects.
    - **Mean |SHAP| bar chart** – global ranking of features by mean absolute SHAP value.

    All SHAP values are also written to a CSV file for downstream analysis.

    When used in :ref:`TrainMLModel`, specify this report under ``reports/models`` in either
    ``selection`` or ``assessment``.

    Reference:

    Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural
    Information Processing Systems, 30. https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html


    **Specification arguments:**

    - n_background_samples (int): number of training samples used as the background distribution for
      KernelExplainer. Ignored for tree and linear models. Default: 100.

    - plot_top_n_features (int): number of top features (by mean |SHAP|) to include in the plots. Set to
      ``null`` to show all features. Default: 25.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_shap_report:
                    SHAPReport:
                        n_background_samples: 100
                        plot_top_n_features: 25

    """

    @classmethod
    def build_object(cls, **kwargs):
        return SHAPReport(
            n_background_samples=kwargs.get("n_background_samples"),
            plot_top_n_features=kwargs.get("plot_top_n_features"),
            name=kwargs.get("name"),
        )

    def __init__(self, n_background_samples: int = None, plot_top_n_features: int = None,
                 train_dataset: Dataset = None, test_dataset: Dataset = None,
                 method: MLMethod = None, result_path: Path = None, name: str = None,
                 hp_setting: HPSetting = None, label: Label = None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method,
                         result_path=result_path, name=name, hp_setting=hp_setting,
                         label=label, number_of_processes=number_of_processes)
        self.n_background_samples = n_background_samples
        self.plot_top_n_features = plot_top_n_features

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        feature_names = self.train_dataset.encoded_data.feature_names
        X_train = self.train_dataset.encoded_data.examples
        X_explain = self.test_dataset.encoded_data.examples if self.test_dataset else X_train

        shap_values = self._compute_shap_values(X_train, X_explain)
        df_shap = pd.DataFrame(shap_values, columns=feature_names)

        mean_abs_shap = df_shap.abs().mean().sort_values(ascending=False)
        n_top = self.plot_top_n_features if self.plot_top_n_features is not None else len(mean_abs_shap)
        top_features = mean_abs_shap.head(n_top).index.tolist()

        top_mean_shap = mean_abs_shap.head(n_top)
        mean_shap_df = pd.DataFrame({"feature": top_mean_shap.index, "mean_abs_shap": top_mean_shap.values})

        table_output = self._write_shap_table(df_shap)
        plots = [
            self._plot_beeswarm(df_shap[top_features], top_features),
            self._plot_mean_shap_bar(mean_shap_df),
        ]

        return ReportResult(
            self.name,
            info=f"SHAP values for {self.method.__class__.__name__} on label '{self.label.name}'",
            output_tables=[table_output],
            output_figures=[p for p in plots if p is not None],
        )

    def _compute_shap_values(self, X_train, X_explain) -> np.ndarray:
        import shap

        n_bg = self.n_background_samples if self.n_background_samples is not None else X_train.shape[0]
        n_bg = min(n_bg, X_train.shape[0])
        idx = np.random.default_rng().choice(X_train.shape[0], n_bg, replace=False)
        background = X_train[idx] if not hasattr(X_train, "iloc") else X_train.iloc[idx]

        explainer = shap.Explainer(self.method.model, background)
        values = explainer(X_explain).values

        # Binary classifiers sometimes return shape (n_samples, n_features, 2) — keep positive class
        if values.ndim == 3:
            values = values[:, :, 1]

        return values

    def _write_shap_table(self, df: pd.DataFrame) -> ReportOutput:
        path = self.result_path / "shap_values.csv"
        df.to_csv(path, index=False)
        return ReportOutput(path, "SHAP values per sample and feature")

    def _plot_beeswarm(self, df_shap: pd.DataFrame, feature_order: list) -> ReportOutput:
        melted = df_shap.melt(var_name="feature", value_name="shap_value")
        melted["feature"] = pd.Categorical(melted["feature"], categories=feature_order[::-1], ordered=True)

        fig = px.strip(
            melted.sort_values("feature"), x="shap_value", y="feature", orientation="h",
            template="plotly_white",
            title=f"SHAP beeswarm — {type(self.method).__name__}",
            labels={"shap_value": "SHAP value", "feature": "Feature"},
        )
        fig.update_traces(marker=dict(opacity=0.4, size=4))

        path = PlotlyUtil.write_image_to_file(fig, self.result_path / "shap_beeswarm.html")
        return ReportOutput(path, "SHAP beeswarm plot")

    def _plot_mean_shap_bar(self, mean_shap: pd.DataFrame) -> ReportOutput:
        fig = px.bar(
            mean_shap.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h",
            template="plotly_white",
            title=f"Mean |SHAP| — {type(self.method).__name__}",
            labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
        )
        fig.update_traces(marker_color=px.colors.sequential.Teal[3])

        path = PlotlyUtil.write_image_to_file(fig, self.result_path / "shap_mean_bar.html")
        return ReportOutput(path, "Mean absolute SHAP values bar chart")

    def check_prerequisites(self) -> bool:
        if not isinstance(self.method, SklearnMethod) and not isinstance(self.method, XGBClassifier):
            logging.warning(
                f"SHAPReport: can only be used with classifiers inheriting SklearnMethod, "
                f"but got {type(self.method).__name__}. Report will not be created."
            )
            return False

        if self.train_dataset is None or self.train_dataset.encoded_data is None:
            logging.warning("SHAPReport: train_dataset with encoded_data is required. Report will not be created.")
            return False

        return True
