import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.PositionHelper import PositionHelper


class AminoAcidFrequencyDistribution(DataReport):
    """
    Generates a barplot showing the relative frequency of each amino acid at each position in the sequences of a dataset.

    **Specification arguments:**

    - imgt_positions (bool): Whether to use IMGT positional numbering or sequence index numbering. When imgt_positions
      is True, IMGT positions are used, meaning sequences of unequal length are aligned according to their IMGT
      positions. By default, imgt_positions is True.

    - relative_frequency (bool): Whether to plot relative frequencies (true) or absolute counts (false) of the
      positional amino acids. By default, relative_frequency is True.

    - split_by_label (bool): Whether to split the plots by a label. If set to true, the Dataset must either contain a
      single label, or alternatively the label of interest can be specified under 'label'. If split_by_label is set to
      true, the percentage-wise frequency difference between classes is plotted additionally. By default,
      split_by_label is False.

    - label (str): if split_by_label is set to True, a label can be specified here.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_aa_freq_report:
                    AminoAcidFrequencyDistribution:
                        relative_frequency: False
                        split_by_label: True
                        label: CMV

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = AminoAcidFrequencyDistribution.__name__
        ParameterValidator.assert_type_and_value(kwargs["imgt_positions"], bool, location, "imgt_positions")
        ParameterValidator.assert_type_and_value(kwargs["relative_frequency"], bool, location, "relative_frequency")
        ParameterValidator.assert_type_and_value(kwargs["split_by_label"], bool, location, "split_by_label")

        if kwargs["label"] is not None:
            ParameterValidator.assert_type_and_value(kwargs["label"], str, location, "label")

            if kwargs["split_by_label"] is False:
                warnings.warn(f"{location}: label is set but split_by_label was False, setting split_by_label to True")
                kwargs["split_by_label"] = True

        return AminoAcidFrequencyDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, imgt_positions: bool = None, relative_frequency: bool = None,
                 split_by_label: bool = None, label: str = None,
                 result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.imgt_positions = imgt_positions
        self.relative_frequency = relative_frequency
        self.split_by_label = split_by_label
        self.label_name = label

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        freq_dist = self._get_plotting_data()

        tables = []
        figures = []

        tables.append(self._write_output_table(freq_dist,
                                               self.result_path / "amino_acid_frequency_distribution.tsv",
                                               name="Table of amino acid frequencies"))

        figures.append(self._safe_plot(freq_dist=freq_dist, plot_callable="_plot_distribution"))

        if self.split_by_label:
            frequency_change = self._compute_frequency_change(freq_dist)

            tables.append(self._write_output_table(frequency_change,
                                                   self.result_path / f"frequency_change.tsv",
                                                   name=f"Frequency change between classes"))
            figures.append(self._safe_plot(frequency_change=frequency_change, plot_callable="_plot_frequency_change"))

        return ReportResult(name=self.name,
                            info="A barplot showing the relative frequency of each amino acid at each position in the sequences of a dataset.",
                            output_figures=[fig for fig in figures if fig is not None],
                            output_tables=[table for table in tables if table is not None])

    def _get_plotting_data(self):
        if isinstance(self.dataset, SequenceDataset):
            plotting_data = self._get_sequence_dataset_plotting_data()

        elif isinstance(self.dataset, ReceptorDataset):
            plotting_data = self._get_receptor_dataset_plotting_data()

        elif isinstance(self.dataset, RepertoireDataset):
            plotting_data = self._get_repertoire_dataset_plotting_data()

        if not self.split_by_label:
            plotting_data.drop(columns=["class"], inplace=True)

        return plotting_data

    def _get_sequence_dataset_plotting_data(self):
        return self._count_dict_per_class_to_df(self._count_aa_frequencies(self._sequence_class_iterator()))

    def _count_dict_per_class_to_df(self, raw_count_dict_per_class):
        result_dfs = []

        for class_name, raw_count_dict in raw_count_dict_per_class.items():
            result_df = self._count_dict_to_df(raw_count_dict)
            result_df["class"] = class_name
            result_dfs.append(result_df)

        return pd.concat(result_dfs)

    def _sequence_class_iterator(self):
        label_name = self._get_label_name()

        for sequence in self.dataset.get_data():
            if self.split_by_label:
                yield (sequence, sequence.get_attribute(label_name))
            else:
                yield (sequence, 0)

    def _get_receptor_dataset_plotting_data(self):
        result_dfs = []

        receptors = self.dataset.get_data()
        chains = next(receptors).get_chains()

        for chain in chains:
            raw_count_dict_per_class = self._count_aa_frequencies(self._chain_class_iterator(chain))

            for class_name, raw_count_dict in raw_count_dict_per_class.items():
                result_df = self._count_dict_to_df(raw_count_dict)
                result_df["chain"] = chain
                result_df["class"] = class_name
                result_dfs.append(result_df)

        return pd.concat(result_dfs)

    def _chain_class_iterator(self, chain):
        label_name = self._get_label_name()

        for receptor in self.dataset.get_data():
            assert chain in receptor.get_chains(), f"{AminoAcidFrequencyDistribution.__name__}: All receptors in the dataset must contain the same chains. Expected {chain} but found {receptor.get_chains()}"

            if self.split_by_label:
                yield (receptor.get_chain(chain), receptor.get_attribute(label_name))
            else:
                yield (receptor.get_chain(chain), 0)

    def _get_repertoire_dataset_plotting_data(self):
        raw_count_dict_per_class = {}
        label_name = self._get_label_name()

        if self.split_by_label:
            class_names = self.dataset.get_metadata([label_name])[label_name]
        else:
            class_names = [0] * self.dataset.get_example_count()

        for repertoire, class_name in zip(self.dataset.get_data(), class_names):
            self._count_aa_frequencies(self._repertoire_class_iterator(repertoire, class_name),
                                       raw_count_dict_per_class)

        return self._count_dict_per_class_to_df(raw_count_dict_per_class)

    def _repertoire_class_iterator(self, repertoire, class_name):
        for sequence in repertoire.get_sequence_objects():
            yield (sequence, class_name)

    def _count_aa_frequencies(self, sequence_class_iterator, raw_count_dict=None):
        raw_count_dict = {} if raw_count_dict is None else raw_count_dict

        for item in sequence_class_iterator:
            sequence, class_name = item
            seq_str = sequence.get_sequence(sequence_type=SequenceType.AMINO_ACID)
            seq_pos = self._get_positions(sequence)

            if class_name not in raw_count_dict:
                raw_count_dict[class_name] = {}

            for aa, pos in zip(seq_str, seq_pos):
                if pos not in raw_count_dict[class_name]:
                    raw_count_dict[class_name][pos] = {legal_aa: 0 for legal_aa in
                                                       EnvironmentSettings.get_sequence_alphabet(
                                                           SequenceType.AMINO_ACID)}

                raw_count_dict[class_name][pos][aa] += 1

        return raw_count_dict

    def _count_dict_to_df(self, raw_count_dict):
        df_dict = {"amino acid": [], "position": [], "count": [], "relative frequency": []}
        for pos in raw_count_dict:
            df_dict["position"].extend([pos] * 20)
            df_dict["amino acid"].extend(list(raw_count_dict[pos].keys()))

            counts = list(raw_count_dict[pos].values())
            total_count_for_pos = sum(counts)

            df_dict["count"].extend(counts)
            df_dict["relative frequency"].extend([count / total_count_for_pos for count in counts])

        return pd.DataFrame(df_dict)

    def _get_positions(self, sequence: ReceptorSequence):
        if self.imgt_positions:
            positions = PositionHelper.gen_imgt_positions_from_length(
                len(sequence.get_sequence(SequenceType.AMINO_ACID)),
                sequence.get_attribute("region_type"))
        else:
            positions = list(range(1, len(sequence.get_sequence(SequenceType.AMINO_ACID)) + 1))

        return [str(pos) for pos in positions]

    def _write_results_table(self, results_table):
        file_path = self.result_path / "amino_acid_frequency_distribution.tsv"

        results_table.to_csv(file_path, index=False)

        return ReportOutput(path=file_path, name="Table of amino acid frequencies")

    def _plot_distribution(self, freq_dist):
        freq_dist.sort_values(by=["amino acid"], ascending=False, inplace=True)
        category_orders = None if "class" not in freq_dist.columns else {"class": sorted(set(freq_dist["class"]))}

        y = "relative frequency" if self.relative_frequency else "count"

        figure = px.bar(freq_dist, x="position", y=y, color="amino acid", text="amino acid",
                        facet_col="class" if "class" in freq_dist.columns else None,
                        facet_row="chain" if "chain" in freq_dist.columns else None,
                        color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                        category_orders=category_orders,
                        labels={"position": "IMGT position" if self.imgt_positions else "Position",
                                "count": "Count",
                                "relative frequency": "Relative frequency",
                                "amino acid": "Amino acid"}, template="plotly_white")
        figure.update_xaxes(categoryorder='array', categoryarray=self._get_position_order(freq_dist["position"]))
        figure.update_layout(showlegend=False, yaxis={'categoryorder': 'category ascending'})

        if self.relative_frequency:
            figure.update_yaxes(tickformat=",.0%", range=[0, 1])

        file_path = self.result_path / "amino_acid_frequency_distribution.html"
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Amino acid frequency distribution")

    def _get_position_order(self, positions):
        return [str(int(pos)) if pos.is_integer() else str(pos) for pos in sorted(set(positions.astype(float)))]

    def _compute_frequency_change(self, freq_dist):
        classes = sorted(set(freq_dist["class"]))
        assert len(classes) == 2, f"{AminoAcidFrequencyDistribution.__name__}: cannot compute frequency change when the number of classes is not 2: {classes}"

        class_a_df = freq_dist[freq_dist["class"] == classes[0]]
        class_b_df = freq_dist[freq_dist["class"] == classes[1]]

        on = ["amino acid", "position"]
        on = on + ["chain"] if "chain" in freq_dist.columns else on

        merged_dfs = pd.merge(class_a_df, class_b_df, on=on, how="outer", suffixes=["_a", "_b"])
        merged_dfs = merged_dfs[(merged_dfs["relative frequency_a"] + merged_dfs["relative frequency_b"]) > 0]

        merged_dfs["frequency_change"] = merged_dfs["relative frequency_a"] - merged_dfs["relative frequency_b"]

        pos_class_a = merged_dfs[merged_dfs["frequency_change"] > 0]
        pos_class_b = merged_dfs[merged_dfs["frequency_change"] < 0]

        pos_class_a["positive_class"] = classes[0]
        pos_class_b["positive_class"] = classes[1]
        pos_class_b["frequency_change"] = 0 - pos_class_b["frequency_change"]

        keep_cols = on + ["frequency_change", "positive_class"]
        pos_class_a = pos_class_a[keep_cols]
        pos_class_b = pos_class_b[keep_cols]

        return pd.concat([pos_class_a, pos_class_b])

    def _plot_frequency_change(self, frequency_change):
        figure = px.bar(frequency_change, x="position", y="frequency_change", color="amino acid", text="amino acid",
                        facet_col="positive_class",
                        facet_row="chain" if "chain" in frequency_change.columns else None,
                        color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                        labels={"position": "IMGT position" if self.imgt_positions else "Position",
                                "positive_class": "Class",
                                "frequency_change": "Difference in relative frequency",
                                "amino acid": "Amino acid"}, template="plotly_white")

        figure.update_xaxes(categoryorder='array', categoryarray=self._get_position_order(frequency_change["position"]))
        figure.update_layout(showlegend=False, yaxis={'categoryorder': 'category ascending'})

        figure.update_yaxes(tickformat=",.0%")

        file_path = self.result_path / "frequency_change.html"
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Frequency difference between amino acid usage in the two classes")

    def _get_label_name(self):
        if self.split_by_label:
            if self.label_name is None:
                return list(self.dataset.get_label_names())[0]
            else:
                return self.label_name
        else:
            return None

    def check_prerequisites(self):
        if self.split_by_label:
            if self.label_name is None:
                if len(self.dataset.get_label_names()) != 1:
                    warnings.warn(f"{AminoAcidFrequencyDistribution.__name__}: ambiguous label: split_by_label was set to True but no label name was specified, and the number of available labels is {len(self.dataset.get_label_names())}: {self.dataset.get_label_names()}. Skipping this report...")
                    return False
            else:
                if self.label_name not in self.dataset.get_label_names():
                    warnings.warn(f"{AminoAcidFrequencyDistribution.__name__}: the specified label name ({self.label_name}) was not available among the dataset labels: {self.dataset.get_label_names()}. Skipping this report...")
                    return False

        return True
