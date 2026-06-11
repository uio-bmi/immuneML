import logging
from pathlib import Path

import numpy as np
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

    For binary classification, the probability for each class is shown. For multiclass
    problems, probabilities for all classes are shown, color-coded by the class the
    probability corresponds to.

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
                 number_of_processes: int = 1, stratify_by: str = None):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method,
                         result_path=result_path, name=name, hp_setting=hp_setting, label=label,
                         number_of_processes=number_of_processes)
        self.stratify_by = stratify_by

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        output_figures = []
        output_tables = []

        for dataset, split in [(self.train_dataset, "train"), (self.test_dataset, "test")]:
            if dataset is not None and dataset.encoded_data is not None:
                fig_output, table_output = self._generate_split_outputs(dataset, split)
                if fig_output is not None:
                    output_figures.append(fig_output)
                if table_output is not None:
                    output_tables.append(table_output)

        return ReportResult(
            self.name,
            info="Predicted class probability distributions grouped by true class label, for training and test data.",
            output_figures=output_figures,
            output_tables=output_tables
        )

    def _generate_split_outputs(self, dataset, split: str):
        try:
            df = self._build_proba_df(dataset)
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: could not compute probabilities for {split} split: {e}")
            return None, None

        csv_path = self.result_path / f"{self.name}_{split}.csv"
        df.to_csv(csv_path, index=False)

        fig = self._make_figure(df, split)
        html_path = self.result_path / f"{self.name}_{split}.html"
        n_examples = len(dataset.encoded_data.labels[self.label.name])
        html_path = PlotlyUtil.write_image_to_file(fig, html_path, n_examples)

        return (
            ReportOutput(html_path, f"Class probability distribution ({split})"),
            ReportOutput(csv_path, f"Class probability data ({split})")
        )

    def _build_proba_df(self, dataset) -> pd.DataFrame:
        encoded_data = dataset.encoded_data
        true_labels = np.array(encoded_data.labels[self.label.name], dtype=str)

        strat_values = None
        if self.stratify_by is not None:
            strat_df = dataset.get_metadata([self.stratify_by], return_df=True)
            strat_values = strat_df[self.stratify_by].astype(str).values

        proba_per_class = self.method.predict_proba(encoded_data, self.label)[self.label.name]

        rows = []
        for class_val, probs in proba_per_class.items():
            for i, p in enumerate(probs):
                row = {
                    "true_class": true_labels[i],
                    "probability_for_class": str(class_val),
                    "probability": float(p)
                }
                if strat_values is not None:
                    row[self.stratify_by] = strat_values[i]
                rows.append(row)
        return pd.DataFrame(rows)

    def _make_figure(self, df: pd.DataFrame, split: str):
        classes_shown = df["probability_for_class"].unique()
        show_color = len(classes_shown) > 1

        fig = px.box(
            df,
            x="true_class",
            y="probability",
            color="probability_for_class" if show_color else None,
            facet_col=self.stratify_by,
            facet_col_wrap=4,
            points="all",
            labels={
                "true_class": "True class",
                "probability": "Predicted probability",
                "probability_for_class": "Probability for class",
                **({}  if self.stratify_by is None else {self.stratify_by: self.stratify_by})
            },
            title=f"Class probability distribution — {split} set ({self.label.name})",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_traces(
            jitter=0.4,
            marker=dict(size=4, opacity=0.6),
            boxmean=True
        )

        return fig

    def check_prerequisites(self) -> bool:
        if not hasattr(self, "result_path") or self.result_path is None:
            logging.warning(f"{self.__class__.__name__} requires a result_path to be set.")
            return False

        if not self.method.can_predict_proba():
            logging.warning(f"{self.__class__.__name__}: the ML method does not support predict_proba "
                            f"and cannot be used with this report.")
            return False

        has_data = False
        for dataset, split in [(self.train_dataset, "train"), (self.test_dataset, "test")]:
            if dataset is not None and dataset.encoded_data is not None:
                has_data = True
            else:
                logging.warning(f"{self.__class__.__name__}: {split} dataset is not encoded; "
                                f"the {split} plot will be skipped.")

        if not has_data:
            logging.warning(f"{self.__class__.__name__}: neither train nor test dataset is encoded.")
            return False

        if self.method is None:
            logging.warning(f"{self.__class__.__name__}: method is not set.")
            return False

        if self.stratify_by is not None:
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
