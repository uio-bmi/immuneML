from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class SequenceCountDistribution(DataReport):
    """
    Generates a histogram of the duplicate counts of the sequences in a dataset.


    **Specification arguments:**

    - split_by_label (bool): Whether to split the plots by a label. If set to true, the Dataset must either contain a single label, or alternatively the label of interest can be specified under 'label'. By default, split_by_label is False.

    - label (str): Optional label for separating the results by color/creating separate plots. Note that this should the name of a valid dataset label.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        my_sld_report:
            SequenceCountDistribution:
                label: disease

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = SequenceCountDistribution.__name__

        ReportUtil.update_split_by_label_kwargs(kwargs, location)

        return SequenceCountDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1,
                 split_by_label: bool = None, label: str = None, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.split_by_label = split_by_label
        self.label_name = label

    def _set_label_name(self):
        if self.split_by_label:
            if self.label_name is None:
                self.label_name = list(self.dataset.get_label_names())[0]
        else:
            self.label_name = None

    def check_prerequisites(self):
        return True

    def _generate(self) -> ReportResult:
        self._set_label_name()

        df = self._get_sequence_counts_df()
        PathBuilder.build(self.result_path)

        output_table = self._write_output_table(df, self.result_path / "sequence_count_distribution.tsv",
                                                name="Duplicate counts of sequences in the dataset")

        report_output_fig = self._safe_plot(df=df, output_written=False)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(name=self.name,
                            info="The sequence count distribution of the dataset.",
                            output_figures=output_figures, output_tables=[output_table])

    def _get_sequence_counts_df(self):
        if isinstance(self.dataset, RepertoireDataset):
            return self._get_repertoire_df()
        elif isinstance(self.dataset, ReceptorDataset) or isinstance(self.dataset, SequenceDataset):
            return self._get_sequence_receptor_df()

    def _get_repertoire_df(self):
        sequence_counts = Counter()

        for repertoire in self.dataset.get_data():
            if self.split_by_label:
                label_class = repertoire.metadata[self.label_name]
            else:
                label_class = None

            repertoire_counter = Counter(repertoire.data.duplicate_count)
            sequence_counts += Counter({(key, label_class): value for key, value in repertoire_counter.items()})

        df = pd.DataFrame({"n_observations": list(sequence_counts.values()),
                           "duplicate_count": [key[0] for key in sequence_counts.keys()]})

        if self.split_by_label:
            df[self.label_name] = [key[1] for key in sequence_counts.keys()]

        return df

    def _get_sequence_receptor_df(self):

        data = self.dataset.data

        try:
            counts = data.duplicate_count
        except AttributeError as e:
            raise AttributeError(
                f"{SequenceCountDistribution.__name__}: SequenceDataset does not contain attribute 'duplicate_count'. "
                f"This report can only be run when sequence counts are available.")

        chains = data.locus.tolist()

        if self.split_by_label:
            label_classes = getattr(data, self.label_name).tolist()
            counter = Counter(zip(counts, chains, label_classes))
        else:
            counter = Counter(zip(counts, chains))

        df = pd.DataFrame({"duplicate_count": [key[0] for key in counter.keys()],
                           "locus": [key[1] for key in counter.keys()],
                           "n_observations": counter.values()})

        if self.split_by_label:
            df[self.label_name] = [key[2] for key in counter.keys()]

        return df

    def _plot(self, df: pd.DataFrame) -> ReportOutput:
        figure = px.bar(df, x="duplicate_count", y="n_observations", barmode="group",
                        color=self.label_name if self.split_by_label else None,
                        facet_col="locus" if "locus" in df.columns and len(set(df["locus"])) > 1 else None,
                        color_discrete_sequence=px.colors.diverging.Tealrose,
                        labels={"n_observations": "Number of observations",
                                "duplicate_count": "Sequence duplicate count"})
        figure.update_layout(template="plotly_white")
        PathBuilder.build(self.result_path)

        file_path = self.result_path / "sequence_count_distribution.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Sequence duplicate count distribution")
