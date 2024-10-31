import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
from pandas import DataFrame

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceLengthDistribution(DataReport):
    """
    Generates a histogram of the lengths of the sequences in a dataset.


    **Specification arguments:**

    - sequence_type (str): Whether to check the length of amino acid or nucleotide sequences; default value is 'amino_acid'

    - split_by_label (bool):  Whether to split the plots by a label. If set to true, the Dataset must either contain a single label, or alternatively the label of interest can be specified under 'label'. By default, split_by_label is False.

    - label (str): Optional label for separating the results by color (individual bars will be shown per class). Note that this should the name of a valid dataset label.

    - as_fraction (bool): When set to True, counts are expressed as a fraction of the total 'group' (defined by chain and optional label). This can be useful when comparing the distribution across labels. When set to False, raw counts are shown. By default, as_fraction is False.

    - plot_type (str): The type of plot to generate; options are 'BAR' and 'LINE'. By default, plot_type is 'BAR'.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_sld_report:
                    SequenceLengthDistribution:
                        sequence_type: amino_acid

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = SequenceLengthDistribution.__name__
        ParameterValidator.assert_sequence_type(kwargs)
        ReportUtil.update_split_by_label_kwargs(kwargs, location)
        kwargs["sequence_type"] = SequenceType[kwargs['sequence_type'].upper()]

        ParameterValidator.assert_type_and_value(kwargs["as_fraction"], bool, location, "as_fraction")
        ParameterValidator.assert_type_and_value(kwargs["plot_type"], str, location, "plot_type")
        ParameterValidator.assert_in_valid_list(kwargs["plot_type"].upper(), ["BAR", "LINE"], location, "plot_type")

        return SequenceLengthDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None, number_of_processes: int = 1,
                 sequence_type: SequenceType = SequenceType.AMINO_ACID, name: str = None,
                 split_by_label: bool = None, label: str = None, as_fraction: bool = None, plot_type: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.batch_size = batch_size
        self.sequence_type = sequence_type
        self.split_by_label = split_by_label
        self.label_name = label
        self.as_fraction = as_fraction
        self.plot_type = plot_type.upper()


    def _set_label_name(self):
        if self.split_by_label:
            if self.label_name is None:
                self.label_name = list(self.dataset.get_label_names())[0]
        else:
            self.label_name = None

    def check_prerequisites(self):
        return True


    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self._set_label_name()

        counts_per_clone, counts_per_sequence = self._get_sequence_lengths_dfs()
        PathBuilder.build(self.result_path)

        clone_table = self._write_output_table(counts_per_clone, self.result_path / 'length_distribution_per_clone.csv', name="Sequence length distribution (aggregrated per clonotype)")
        sequence_table = self._write_output_table(counts_per_sequence, self.result_path / 'length_distribution_per_sequence.csv', name="Sequence length distribution (considering duplicate counts)")
        tables = [clone_table, sequence_table]

        if self.as_fraction:
            counts_per_clone_fr = self._convert_counts_to_fraction(counts_per_clone, "number_of_clones")
            counts_per_sequence_fr = self._convert_counts_to_fraction(counts_per_sequence, "number_of_sequences")

            tables.append(self._write_output_table(counts_per_clone_fr, self.result_path / 'length_distribution_per_clone_as_fraction.csv', name="Sequence length distribution (aggregrated per clonotype) as a fraction"))
            tables.append(self._write_output_table(counts_per_sequence_fr, self.result_path / 'length_distribution_per_sequence_as_fraction.csv', name="Sequence length distribution (considering duplicate counts) as a fraction"))

            clone_fig = self._safe_plot(df=counts_per_clone_fr,
                                        file_path=self.result_path / "length_distribution_per_clone.html",
                                        y="number_of_clones_as_fraction",
                                        name="Sequence length distribution (aggregrated per clonotype)")
            sequence_fig = self._safe_plot(df=counts_per_sequence_fr,
                                           file_path=self.result_path / "length_distribution_per_sequence.html",
                                           y="number_of_sequences_as_fraction",
                                           name="Sequence length distribution (considering duplicate counts)")

        else:
            clone_fig = self._safe_plot(df=counts_per_clone,
                                        file_path=self.result_path / "length_distribution_per_clone.html",
                                        y="number_of_clones",
                                        name="Sequence length distribution (aggregrated per clonotype)")
            sequence_fig = self._safe_plot(df=counts_per_sequence,
                                           file_path=self.result_path / "length_distribution_per_sequence.html",
                                           y="number_of_sequences",
                                           name="Sequence length distribution (considering duplicate counts)")

        output_figures = [fig for fig in [clone_fig, sequence_fig] if fig is not None]
        output_figures = None if len(output_figures) == 0 else output_figures

        return ReportResult(name=self.name,
                            info=f"A histogram of the lengths of the sequences in a dataset.",
                            output_figures=output_figures, output_tables=tables)

    def _convert_counts_to_fraction(self, counts_table, counts_col):
        group_cols = []
        if "chain" in counts_table.columns: # todo should repertoire option be added?
            group_cols.append("chain")

        if self.split_by_label:
            group_cols.append(self.label_name)

        total_col = f"total_{counts_col}_in_" + "_".join(group_cols)

        if len(group_cols) > 0:
            total_per_group = counts_table.groupby(group_cols)[counts_col].sum().reset_index()
            total_per_group.rename(columns={counts_col: total_col}, inplace=True)
            counts_table_fr = counts_table.merge(total_per_group, on=group_cols)
            counts_table_fr[f"{counts_col}_as_fraction"] = counts_table_fr[counts_col] / counts_table_fr[total_col]

        else:
            counts_table_fr = counts_table.copy()
            counts_table_fr[f"total_{counts_col}"] = counts_table_fr[counts_col].sum()
            counts_table_fr[f"{counts_col}_as_fraction"] = counts_table_fr[counts_col] / counts_table_fr[counts_col].sum()

        return counts_table_fr

    def _get_sequence_lengths_dfs(self) -> DataFrame:
        if isinstance(self.dataset, RepertoireDataset):
            return self._get_sequence_lengths_df_repertoire_dataset()
        elif isinstance(self.dataset, SequenceDataset):
            return self._get_length_dfs_sequence_dataset()
        elif isinstance(self.dataset, ReceptorDataset):
            raise NotImplementedError()
            return self._get_length_dfs_receptor_dataset()

    def _get_sequence_lengths_df_repertoire_dataset(self):
        sequence_lengths = Counter()

        for repertoire in self.dataset.get_data(self.batch_size):
            seq_lengths = self._count_in_repertoire(repertoire)
            sequence_lengths += seq_lengths

        return pd.DataFrame({"counts": list(sequence_lengths.values()),
                             'sequence_lengths': list(sequence_lengths.keys())})


    def _get_length_dfs_sequence_dataset(self):
        counts_per_clone = Counter([(len(sequence.get_sequence(self.sequence_type)), sequence.get_attribute(str(self.label_name)))
                                        for sequence in self.dataset.get_data()])

        counts_per_sequence = Counter(((len(sequence.get_sequence(self.sequence_type)), sequence.get_attribute(str(self.label_name)))
                                           for sequence in self.dataset.get_data()
                                           for i in range(0, sequence.get_duplicate_count())))

        clones_df = pd.DataFrame({"number_of_clones": list(counts_per_clone.values()),
                              "sequence_lengths": [key[0] for key in counts_per_clone.keys()]})

        seqs_df = pd.DataFrame({"number_of_sequences": list(counts_per_sequence.values()),
                              "sequence_lengths": [key[0] for key in counts_per_sequence.keys()]})

        if self.split_by_label:
            clones_df[self.label_name] = [key[1] for key in counts_per_clone.keys()]
            seqs_df[self.label_name] = [key[1] for key in counts_per_sequence.keys()]

        return (clones_df.sort_values(by="sequence_lengths"),
                seqs_df.sort_values(by="sequence_lengths"))

    def _get_dataset_chains(self):
        return next(self.dataset.get_data()).get_chains()

    def _get_length_dfs_receptor_dataset(self):
        chains = self._get_dataset_chains()

        clone_counter = Counter()
        sequence_counter = Counter()

        for receptor in self.dataset.get_data():
            assert receptor.get_chains() == chains, f"{SequenceLengthDistribution.__name__}: All receptors must be of the same type. Found different chain types: {chains} and {receptor.get_chains}"

            for chain_name in receptor.get_chains():
                chain = getattr(receptor, chain_name)
                clone_counter.update([(len(chain.get_sequence(self.sequence_type)), chain.get_attribute(str(self.label_name)), chain_name)])
                sequence_counter.update([(len(chain.get_sequence(self.sequence_type)), chain.get_attribute(str(self.label_name)), chain_name)] * chain.get_duplicate_count())

        clones_df = pd.DataFrame({"number_of_clones": list(clone_counter.values()),
                                  "sequence_lengths": [key[0] for key in clone_counter.keys()],
                                  "chain": [key[2] for key in clone_counter.keys()]})

        seqs_df = pd.DataFrame({"number_of_sequences": list(sequence_counter.values()),
                                "sequence_lengths": [key[0] for key in sequence_counter.keys()],
                                "chain": [key[2] for key in sequence_counter.keys()]})

        if self.split_by_label:
            clones_df[self.label_name] = [key[1] for key in clone_counter.keys()]
            seqs_df[self.label_name] = [key[1] for key in sequence_counter.keys()]

        return (clones_df.sort_values(by="sequence_lengths"),
                seqs_df.sort_values(by="sequence_lengths"))

    def _count_in_repertoire(self, repertoire: Repertoire) -> Counter:
        c = Counter([len(sequence.get_sequence(self.sequence_type)) for sequence in repertoire.sequences])
        return c

    def _plot(self, df: pd.DataFrame, file_path, y, name) -> ReportOutput:
        plot_kwargs = {"x": "sequence_lengths", "y": y, "color": self.label_name,
                        "facet_col": "chain" if isinstance(self.dataset, ReceptorDataset) else None,
                        "color_discrete_sequence": px.colors.diverging.Tealrose,
                        "labels":{
                            "number_of_clones": "Number of clones",
                            "number_of_sequences": "Number of sequences",
                            "number_of_clones_as_fraction": "Fraction of clones",
                            "number_of_sequences_as_fraction": "Fraction of sequences",
                        }}

        if self.plot_type == "LINE":
            figure = px.line(df, **plot_kwargs)
        else:
            figure = px.bar(df, barmode="group", **plot_kwargs)

        figure.update_layout(xaxis=dict(tickmode='array', tickvals=df["sequence_lengths"]),
                             yaxis=dict(tickmode='array', tickvals=df[y]),
                             yaxis2=dict(tickmode='array', tickvals=df[y]),
                             template="plotly_white")

        PathBuilder.build(self.result_path)

        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=name)

