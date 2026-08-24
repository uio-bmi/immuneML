import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.environment.Label import Label


class ClassProbabilityDistribution(MLReport):
    """
    A report that plots the distribution of predicted class probabilities for each example,
    grouped by the true class label. Individual data points are shown alongside box plots.
    Two plots are generated: one for training data and one for test data.

    If :py:obj:`~immuneML.environment.Label.Label` has a positive class set,
    it is used to consistently order and color the classes (with the positive class always shown
    last / in the same color) so that the plots for train and test data remain comparable.

    For binary classification, only the probability of the positive class is shown, grouped by true
    class (one box per true class) - for a good classifier, this is low for the negative class and
    high for the positive class. For multiclass problems, probabilities for all classes are shown,
    color-coded by the true class of the example.

    **Specification arguments:**

    - stratify_by (str): optional name of a metadata label to use as facets in the plot.
      Each unique value of this label will become a separate column of subplots, allowing
      comparison of probability distributions across subgroups (e.g., batches, cohorts).
      Default is None.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_prob_report:
                    ClassProbabilityDistribution:
                        stratify_by: batch

    """

    @classmethod
    def build_object(cls, **kwargs):
        return ClassProbabilityDistribution(**kwargs)

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting: HPSetting = None, label: Label = None,
                 number_of_processes: int = 1, stratify_by: str = None, train_predictions_path: Path = None,
                 test_predictions_path: Path = None):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method,
                         result_path=result_path, name=name, hp_setting=hp_setting, label=label,
                         number_of_processes=number_of_processes, train_predictions_path=train_predictions_path,
                         test_predictions_path=test_predictions_path)
        self.stratify_by = stratify_by

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        output_figures = []
        splits = [(self.train_dataset, self.train_predictions_path, "train"),
                  (self.test_dataset, self.test_predictions_path, "test")]

        for dataset, predictions_path, split in splits:
            if predictions_path is not None and Path(predictions_path).is_file():
                fig_output = self._generate_split_output(dataset, predictions_path, split)
                if fig_output is not None:
                    output_figures.append(fig_output)
            else:
                logging.warning(f"{self.__class__.__name__}: no stored predictions found for {split} split "
                                f"({predictions_path}); the {split} plot will be skipped.")

        return ReportResult(
            self.name,
            info="Predicted class probability distributions grouped by true class label, for training and test data.",
            output_figures=output_figures
        )

    def _generate_split_output(self, dataset, predictions_path: Path, split: str):
        try:
            df, n_examples = self._build_proba_df(dataset, predictions_path)
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not read stored predictions for {split} split: {e}")
            return None

        fig = self._make_figure(df, split)
        html_path = self.result_path / f"{self.name}_{split}.html"
        html_path = PlotlyUtil.write_image_to_file(fig, html_path, n_examples)
        return ReportOutput(html_path, f"Class probability distribution ({split})")

    @staticmethod
    def _find_id_col(raw_df: pd.DataFrame):
        return next((col for col in ["example_id", "example_ids"] if col in raw_df.columns), None)

    def _get_available_proba_cols(self, raw_df: pd.DataFrame, classes: list, predictions_path: Path) -> dict:
        proba_cols = {cls: f"{self.label.name}_{cls}_proba" for cls in classes}
        available = {cls: col for cls, col in proba_cols.items() if col in raw_df.columns}
        missing = [cls for cls in classes if cls not in available]
        if missing:
            logging.warning(f"{self.__class__.__name__}: no stored probabilities found for class(es) {missing} "
                            f"in {predictions_path}; they will be skipped.")
        return available

    def _build_positive_class_probability_df(self, raw_df: pd.DataFrame, true_class: pd.Series, available_proba_cols: dict) -> pd.DataFrame:
        pos_class = str(self.label.positive_class)
        assert pos_class in available_proba_cols, \
            f"{self.__class__.__name__}: stored probabilities for the positive class ({pos_class}) of label " \
            f"{self.label.name} are required, found: {list(available_proba_cols.keys())}."
        return pd.DataFrame({"true_class": true_class, "probability": raw_df[available_proba_cols[pos_class]]})

    @staticmethod
    def _build_multiclass_df(raw_df: pd.DataFrame, true_class_col: str, id_col: str, available_proba_cols: dict) -> pd.DataFrame:
        id_vars = [true_class_col] + ([id_col] if id_col is not None else [])
        df = raw_df[id_vars + list(available_proba_cols.values())].melt(
            id_vars=id_vars, value_vars=list(available_proba_cols.values()),
            var_name="_proba_col", value_name="probability")
        col_to_class = {col: cls for cls, col in available_proba_cols.items()}
        df["probability_for_class"] = df["_proba_col"].map(col_to_class)
        df["true_class"] = df[true_class_col].astype(str)
        return df.drop(columns=["_proba_col", true_class_col])

    def _add_stratify_column(self, df: pd.DataFrame, dataset, id_col: str) -> pd.DataFrame:
        if self.stratify_by is None or dataset is None or "_id" not in df.columns:
            return df.drop(columns=["_id"], errors="ignore")

        strat_df = dataset.get_metadata([self.stratify_by], return_df=True)
        strat_by_id = dict(zip([str(eid) for eid in dataset.get_example_ids()], strat_df[self.stratify_by].astype(str)))
        df[self.stratify_by] = df["_id"].map(strat_by_id)
        return df.drop(columns=["_id"])

    def _build_proba_df(self, dataset, predictions_path: Path) -> tuple:
        raw_df = pd.read_csv(predictions_path)

        true_class_col = f"{self.label.name}_true_class"
        assert true_class_col in raw_df.columns, \
            f"{self.__class__.__name__}: column '{true_class_col}' was not found in stored predictions {predictions_path}."

        id_col = self._find_id_col(raw_df)
        classes_positive_last = [str(cls) for cls in self.label.values]
        available_proba_cols = self._get_available_proba_cols(raw_df, classes_positive_last, predictions_path)
        is_binary = len(classes_positive_last) == 2

        if is_binary:
            df = self._build_positive_class_probability_df(raw_df, raw_df[true_class_col].astype(str), available_proba_cols)
            if id_col is not None:
                df["_id"] = raw_df[id_col].astype(str)
        else:
            df = self._build_multiclass_df(raw_df, true_class_col, id_col, available_proba_cols)
            if id_col is not None:
                df = df.rename(columns={id_col: "_id"})

        df = self._add_stratify_column(df, dataset, id_col)

        columns = ["true_class"] + (["probability_for_class"] if "probability_for_class" in df.columns else []) + \
                  ["probability"] + ([self.stratify_by] if self.stratify_by is not None else [])
        return df[columns], len(raw_df)

    def _plot_axis_config(self, is_multiclass: bool) -> tuple:
        classes_order = [str(cls) for cls in self.label.values]
        x = "probability_for_class" if is_multiclass else "true_class"
        color = "true_class"
        category_orders = {"true_class": classes_order}
        if is_multiclass:
            category_orders["probability_for_class"] = classes_order
        positive_class_label = f"Predicted probability of positive class ({self.label.positive_class})"
        labels = {
            "true_class": "True class",
            "probability": "Predicted probability" if is_multiclass else positive_class_label,
            "probability_for_class": "Probability for class",
            **({} if self.stratify_by is None else {self.stratify_by: self.stratify_by})
        }
        return x, color, category_orders, labels

    def _make_figure(self, df: pd.DataFrame, split: str):
        is_multiclass = "probability_for_class" in df.columns
        x, color, category_orders, labels = self._plot_axis_config(is_multiclass)

        fig = px.box(
            df,
            x=x,
            y="probability",
            color=color,
            facet_col=self.stratify_by,
            facet_col_wrap=4,
            points="all",
            category_orders=category_orders,
            labels=labels,
            title=f"Class probability distribution — {split} set ({self.label.name})",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.6), boxmean=True)
        fig.update_yaxes(range=[-0.05, 1.05])
        return fig

    def check_prerequisites(self) -> bool:
        if not hasattr(self, "result_path") or self.result_path is None:
            logging.warning(f"{self.__class__.__name__} requires a result_path to be set.")
            return False

        if self.label is None:
            logging.warning(f"{self.__class__.__name__}: label is not set.")
            return False

        if not self._has_stored_predictions():
            logging.warning(f"{self.__class__.__name__}: no stored predictions are available for either train or test data.")
            return False

        return self._stratify_by_is_valid()

    def _has_stored_predictions(self) -> bool:
        has_data = False
        for predictions_path, split in [(self.train_predictions_path, "train"), (self.test_predictions_path, "test")]:
            if predictions_path is not None and Path(predictions_path).is_file():
                has_data = True
            else:
                logging.warning(f"{self.__class__.__name__}: stored predictions for {split} data are not available "
                                f"({split}_predictions_path is not set or the file does not exist); "
                                f"the {split} plot will be skipped.")
        return has_data

    def _stratify_by_is_valid(self) -> bool:
        if self.stratify_by is None:
            return True

        for dataset, split in [(self.train_dataset, "train"), (self.test_dataset, "test")]:
            if dataset is None:
                continue
            try:
                available = dataset.get_metadata([self.stratify_by], return_df=True).columns.tolist()
                if self.stratify_by not in available:
                    logging.warning(f"{self.__class__.__name__}: stratify_by label '{self.stratify_by}' "
                                    f"not found in {split} dataset metadata.")
                    return False
            except Exception as e:
                logging.warning(f"{self.__class__.__name__}: could not retrieve stratify_by label "
                                f"'{self.stratify_by}' from {split} dataset: {e}")
                return False
        return True