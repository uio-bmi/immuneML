import warnings
from ast import Param
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class LabelBalance(DataReport):
    """
    Plots the balance of class labels (number of elements associated with each class).

    For Sequence- and ReceptorDatasets, a separate figure is made for the number of clonotypes and number of duplicates if duplicate counts are available.


    Specification arguments:

        labels (list): Labels to apply the report to. If no labels are specified, all labels associated with the class are used.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_vj_gene_report:
                    VJGeneDistribution:
                        label: ag_binding

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = LabelBalance.__name__

        ParameterValidator.assert_type_and_value(kwargs["labels"], list, location, "labels")

        return LabelBalance(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1,
                 name: str = None, labels: list = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.labels = labels if labels is not None else []

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self._set_labels(self.dataset)

        self.dataset.get_metadata(self.labels)
        assert len(self.labels) > 0, "LabelBalance: report was specified but no valid labels were found."

        tables = []
        figures = []

        for label_name in self.labels:
            if type(self.dataset) == RepertoireDataset:
                raw_data = self.dataset.get_metadata([label_name])
            else:
                raw_data = self.dataset.get_metadata([label_name, "duplicate_count"])

            clones_df = self._get_clones_df(raw_data, label_name)

            tables.append(self._write_output_table(clones_df,
                                                   file_path=self.result_path/f"{label_name}_balance_n_clones.tsv",
                                                   name=f"Class balance of label {label_name} (number of clones)"))
            figures.append(self._plot_label_balance(clones_df, label_name, count_col="n_clones",
                                                    file_path=self.result_path/f"{label_name}_balance_n_clones.html",
                                                    name=f"Class balance of label {label_name} (number of clones)"))

            if "duplicate_count" in raw_data and len(set(raw_data["duplicate_count"])) > 1:
                dups_df = self._get_duplicates_df(raw_data, label_name)

                if dups_df is not None:
                    tables.append(self._write_output_table(dups_df,
                                                           file_path=self.result_path / f"{label_name}_balance_n_duplicates.tsv",
                                                           name=f"Class balance of label {label_name} (total number of duplicates)"))
                    figures.append(self._plot_label_balance(dups_df, label_name, count_col="n_duplicates",
                                                            file_path=self.result_path / f"{label_name}_balance_n_duplicates.html",
                                                            name=f"Class balance of label {label_name} (total number of duplicates)"))

        return ReportResult(name=self.name,
                            info=f"Label balance per label ({', '.join(self.labels)})",
                            output_figures=[fig for fig in figures if fig is not None],
                            output_tables=[table for table in tables if table is not None])

    def _get_clones_df(self, raw_data, label_name):
        clones_df = pd.DataFrame(Counter(raw_data[label_name]).items(), columns=[label_name, "n_clones"])

        if type(self.dataset) == ReceptorDataset:
            clones_df["n_clones"] = clones_df["n_clones"]

        return clones_df

    def _get_duplicates_df(self, raw_data, label_name):
        duplicate_counter = Counter()
        for label_class, duplicate_count in zip(raw_data[label_name], raw_data["duplicate_count"]):
            if type(duplicate_count) == str:
                warnings.warn(f"LabelBalance: unable to compute number of duplicates per label class for dataset {self.dataset.name}. "
                              f"This could happen when the duplicate counts differ for the 2 chains of a ReceptorDataset. "
                              f"This figure will be omitted...")
                return None

            duplicate_counter += {label_class: duplicate_count}

        dups_df = pd.DataFrame(duplicate_counter.items(), columns=[label_name, "n_duplicates"])

        if type(self.dataset) == ReceptorDataset:
            dups_df["n_clones"] = dups_df["n_duplicates"]

        return dups_df

    def _set_labels(self, dataset):
        if len(self.labels) == 0:
            self.labels = dataset.get_label_names()



    def _plot_label_balance(self, df, label_name, count_col, file_path, name):
        figure = px.bar(df, x=label_name, y=count_col, color=label_name,
                        labels={"n_clones": "Number of clones",
                                "n_duplicates": "Total number of duplicates (summed across clones)"},
                        color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(template="plotly_white",
                             barmode="relative")

        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=name)
