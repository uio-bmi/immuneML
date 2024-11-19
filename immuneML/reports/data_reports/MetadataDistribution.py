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


class MetadataDistribution(DataReport):
    """
    Plots a bar plot showing the distribution of class labels (number of elements associated with each class) or any other metadata field (e.g., locus for sequence- and receptor  datasets).

    For Sequence- and ReceptorDatasets, a separate figure is made for the number of clonotypes (= number of sequence/receptor examples in the dataset) and number of duplicates if duplicate counts are available.


    **Specification arguments:**

        fields (list): metadat fields to apply the report to. If no fields are specified, all *labels* associated with the class are used.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_label_report:
                    MetadataDistribution:
                        fields: [label_name_1, label_name_2] # label names or standard metadata fields may be specified here
                my_locus_report:
                    MetadataDistribution:
                        fields: [locus]

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = MetadataDistribution.__name__

        ParameterValidator.assert_type_and_value(kwargs["fields"], list, location, "fields")

        return MetadataDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1,
                 name: str = None, fields: list = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.fields = fields if fields is not None else []

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self._set_fields(self.dataset)

        tables = []
        figures = []

        for field_name in self.fields:
            try:
                label_tables, label_figures = self._get_result_for_label(field_name)
                tables.extend(label_tables)
                figures.extend(label_figures)
            except Exception as e:
                warnings.warn(f"{MetadataDistribution.__name__}: An error occurred when attempting to compute the distribution for "
                              f"metadata '{field_name}' on dataset {self.dataset.name}: {e}. Skipping this metadata field...")

        return ReportResult(name=self.name,
                            info=f"Label balance per label ({', '.join(self.fields)})",
                            output_figures=[fig for fig in figures if fig is not None],
                            output_tables=[table for table in tables if table is not None])

    def _get_result_for_label(self, field_name):
        tables = []
        figures = []

        if type(self.dataset) == RepertoireDataset:
            raw_data = self.dataset.get_metadata([field_name])
        else:
            raw_data = self.dataset.get_metadata([field_name, "duplicate_count"])

        clones_df = self._get_clones_df(raw_data, field_name)

        tables.append(self._write_output_table(clones_df,
                                               file_path=self.result_path / f"{field_name}_balance_n_clones.tsv",
                                               name=f"Class balance of {field_name} (number of clones)"))
        figures.append(self._plot_metadata_distribution(clones_df, field_name, count_col="n_clones",
                                                        file_path=self.result_path / f"{field_name}_balance_n_clones.html",
                                                        name=f"Class balance of {field_name} (number of clones)"))

        if "duplicate_count" in raw_data and len(set(raw_data["duplicate_count"])) > 1:
            dups_df = self._get_duplicates_df(raw_data, field_name)

            if dups_df is not None:
                tables.append(self._write_output_table(dups_df,
                                                       file_path=self.result_path / f"{field_name}_balance_n_duplicates.tsv",
                                                       name=f"Class balance of {field_name} (total number of duplicates)"))
                figures.append(self._plot_metadata_distribution(dups_df, field_name, count_col="n_duplicates",
                                                                file_path=self.result_path / f"{field_name}_balance_n_duplicates.html",
                                                                name=f"Class balance of {field_name} (total number of duplicates)"))

        return tables, figures

    def _get_clones_df(self, raw_data, field_name):
        clones_df = pd.DataFrame(Counter(raw_data[field_name]).items(), columns=[field_name, "n_clones"])

        if type(self.dataset) == ReceptorDataset:
            clones_df["n_clones"] = clones_df["n_clones"]

        return clones_df

    def _get_duplicates_df(self, raw_data, field_name):
        duplicate_counter = Counter()
        for field_value, duplicate_count in zip(raw_data[field_name], raw_data["duplicate_count"]):
            if type(duplicate_count) == str:
                warnings.warn(f"{MetadataDistribution.__name__}: unable to compute number of duplicates per example for dataset {self.dataset.name}. "
                              f"This could happen when the duplicate counts differ for the 2 chains of a ReceptorDataset. "
                              f"This figure will be omitted...")
                return None

            duplicate_counter += {field_value: duplicate_count}

        dups_df = pd.DataFrame(duplicate_counter.items(), columns=[field_name, "n_duplicates"])

        if type(self.dataset) == ReceptorDataset:
            dups_df["n_clones"] = dups_df["n_duplicates"]

        return dups_df

    def _set_fields(self, dataset):
        if len(self.fields) == 0:
            self.fields = dataset.get_label_names()

            assert len(self.fields) > 0, f"{MetadataDistribution.__name__}: report was specified but no valid labels were found."

    def _plot_metadata_distribution(self, df, field_name, count_col, file_path, name):
        figure = px.bar(df, x=field_name, y=count_col, color=field_name,
                        labels={"n_clones": "Number of clones",
                                "n_duplicates": "Total number of duplicates (summed across clones)"},
                        color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(template="plotly_white",
                             barmode="relative")

        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=name)
