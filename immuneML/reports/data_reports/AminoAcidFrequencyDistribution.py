import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model import bnp_util
from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.PositionHelper import PositionHelper


class AminoAcidFrequencyDistribution(DataReport):
    """
    Generates a barplot showing the relative frequency of each amino acid at each position in the sequences of a dataset.

    Example output:

    .. image:: ../../_static/images/reports/amino_acid_frequency.png
       :alt: Amino acid frequency
       :width: 800

    .. image:: ../../_static/images/reports/amino_acid_frequency_change.png
       :alt: Amino acid frequency change
       :width: 800

    **Specification arguments:**

    - alignment (str): Alignment style for aligning sequences of different lengths. Options are as follows:

      - CENTER: center-align sequences of different lengths. The middle amino acid of any sequence be labelled position 0. By default, alignment is CENTER.

      - LEFT: left-align sequences of different lengths, starting at 0.

      - RIGHT: right align sequences of different lengths, ending at 0 (counting towards negative numbers).

      - IMGT: align sequences based on their IMGT positional numbering, considering the sequence region_type (IMGT_CDR3 or IMGT_JUNCTION).
        The main difference between CENTER and IMGT is that IMGT aligns the first and last amino acids, adding gaps in the middle,
        whereas CENTER aligns the middle of the sequences, padding with gaps at the start and end of the sequence.
        When region_type is IMGT_JUNCTION, the IMGT positions run from 104 (conserved C) to 118 (conserved W/F). When IMGT_CDR3 is used, these positions are 105 to 117.
        For long CDR3 sequences, additional numbers are added in between IMGT positions 111 and 112.
        See the official IMGT documentation for more details: https://www.imgt.org/IMGTScientificChart/Numbering/CDR3-IMGTgaps.html

    - relative_frequency (bool): Whether to plot relative frequencies (true) or absolute counts (false) of the
      positional amino acids. Note that when sequences are of different length, setting relative_frequency to True will
      produce different results depending on the alignment type, as some positions are only covered by the longest sequences.
      By default, relative_frequency is False.

    - split_by_label (bool): Whether to split the plots by a label. If set to true, the Dataset must either contain a
      single label, or alternatively the label of interest can be specified under 'label'. If split_by_label is set to
      true, the percentage-wise frequency difference between classes is plotted additionally. By default,
      split_by_label is False.

    - label (str): if split_by_label is set to True, a label can be specified here.

    - region_type (str): which part of the sequence to check; e.g., IMGT_CDR3

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
                        region_type: IMGT_CDR3

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = AminoAcidFrequencyDistribution.__name__

        if "imgt_positions" in kwargs:
            raise ValueError(
                f"{location}: parameter 'imgt_positions' is deprecated. For 'imgt_positions: True', use 'alignment: IMGT'. For 'imgt_positions: False', use any other alignment option (CENTER/LEFT/RIGHT).")

        ParameterValidator.assert_type_and_value(kwargs["alignment"], str, location, "alignment")
        ParameterValidator.assert_in_valid_list(kwargs["alignment"].upper(), ["IMGT", "LEFT", "RIGHT", "CENTER"],
                                                location, "alignment")
        ParameterValidator.assert_type_and_value(kwargs["relative_frequency"], bool, location, "relative_frequency")
        ParameterValidator.assert_type_and_value(kwargs["split_by_label"], bool, location, "split_by_label")
        ParameterValidator.assert_region_type(kwargs, location)

        ReportUtil.update_split_by_label_kwargs(kwargs, location)

        return AminoAcidFrequencyDistribution(**{**kwargs, 'region_type': RegionType[kwargs['region_type'].upper()]})

    def __init__(self, dataset: Dataset = None, alignment: bool = None, relative_frequency: bool = None,
                 split_by_label: bool = None, label: str = None, region_type: RegionType = RegionType.IMGT_CDR3,
                 result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.alignment = alignment
        self.relative_frequency = relative_frequency
        self.split_by_label = split_by_label
        self.label_name = label
        self.region_type = region_type

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self.label_name = self._get_label_name()

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
                            info="A barplot showing the relative frequency of each amino acid at each position in "
                                 "the sequences of a dataset.",
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
            plotting_data.drop(columns=[self.label_name], inplace=True)

        return plotting_data

    def _get_sequence_dataset_plotting_data(self):
        return self._count_dict_per_class_to_df(self._count_aa_frequencies(self.dataset.data, class_name=None))

    def _count_dict_per_class_to_df(self, raw_count_dict_per_class):
        result_dfs = []

        for class_name, raw_count_dict in raw_count_dict_per_class.items():
            result_df = self._count_dict_to_df(raw_count_dict)
            result_df[self.label_name] = class_name
            result_dfs.append(result_df)

        return pd.concat(result_dfs)

    def _get_receptor_dataset_plotting_data(self):
        result_dfs = []

        data = self.dataset.data
        chains = list(set(data.locus.tolist()))

        for chain in chains:
            mask = [el == chain for el in data.locus.tolist()]
            raw_count_dict_per_class = self._count_aa_frequencies(data[mask])

            for class_name, raw_count_dict in raw_count_dict_per_class.items():
                result_df = self._count_dict_to_df(raw_count_dict)
                result_df["locus"] = chain
                result_df[self.label_name] = class_name
                result_dfs.append(result_df)

        return pd.concat(result_dfs)

    def _get_repertoire_dataset_plotting_data(self):
        raw_count_dict_per_class = {}
        label_name = self._get_label_name()

        if self.split_by_label:
            class_names = self.dataset.get_metadata([label_name])[label_name]
        else:
            class_names = [0] * self.dataset.get_example_count()

        for repertoire, class_name in zip(self.dataset.get_data(), class_names):
            self._count_aa_frequencies(repertoire.data, raw_count_dict_per_class, class_name)

        return self._count_dict_per_class_to_df(raw_count_dict_per_class)

    def _count_aa_frequencies(self, data: AIRRSequenceSet, raw_count_dict=None, class_name: str = None):
        raw_count_dict = {} if raw_count_dict is None else raw_count_dict

        for item in data.to_iter():
            sequence = getattr(item, bnp_util.get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID))
            seq_pos = self._get_positions(len(sequence))

            if self.split_by_label and self.label_name is not None:
                if class_name is None:
                    cls_name = getattr(item, self.label_name)
                else:
                    cls_name = class_name
            else:
                cls_name = 0

            if cls_name not in raw_count_dict:
                raw_count_dict[cls_name] = {}

            for aa, pos in zip(sequence, seq_pos):
                if pos not in raw_count_dict[cls_name]:
                    raw_count_dict[cls_name][pos] = {legal_aa: 0 for legal_aa in
                                                     EnvironmentSettings.get_sequence_alphabet(
                                                         SequenceType.AMINO_ACID)}

                raw_count_dict[cls_name][pos][aa] += 1

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

    def _get_positions(self, sequence_length: int):
        if self.alignment == 'IMGT':
            positions = PositionHelper.gen_imgt_positions_from_length(sequence_length, self.region_type)
        elif self.alignment == 'LEFT':
            positions = list(range(1, sequence_length + 1))
        elif self.alignment == "RIGHT":
            positions = list(range(-sequence_length + 2, 1))
        else:  # self.alignment == "CENTER
            positions = list(range(1, sequence_length + 1))
            positions = [pos - round(max(positions) / 2) for pos in positions]

        return [str(pos) for pos in positions]

    def _write_results_table(self, results_table):
        file_path = self.result_path / "amino_acid_frequency_distribution.tsv"

        results_table.to_csv(file_path, index=False)

        return ReportOutput(path=file_path, name="Table of amino acid frequencies")

    def _plot_distribution(self, freq_dist):
        freq_dist.sort_values(by=["amino acid"], ascending=False, inplace=True)
        category_orders = None if self.label_name not in freq_dist.columns \
            else {self.label_name: sorted(set(freq_dist[self.label_name]))}

        y = "relative frequency" if self.relative_frequency else "count"

        figure = px.bar(freq_dist, x="position", y=y, color="amino acid", text="amino acid",
                        facet_col=self.label_name if self.label_name in freq_dist.columns else None,
                        facet_row="locus" if "locus" in freq_dist.columns else None,
                        color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                        category_orders=category_orders,
                        labels={"position": "IMGT position" if self.alignment == "IMGT" else "Position",
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
        if self.alignment == "IMGT":
            if min(positions) == "105" and max(positions) == "117":
                return PositionHelper.gen_imgt_positions_from_cdr3_length(len(set(positions)))
            elif min(positions) == "104" and max(positions) == "118":
                return ["104"] + PositionHelper.gen_imgt_positions_from_cdr3_length(len(set(positions)) - 2) + ["118"]
        else:
            return [str(pos) for pos in sorted(set(positions.astype(int)))]

    def _compute_frequency_change(self, freq_dist):
        classes = sorted(set(freq_dist[self.label_name]))
        assert len(classes) == 2, \
            (f"{AminoAcidFrequencyDistribution.__name__}: cannot compute frequency change when the number of "
             f"classes is not 2: {classes}")

        class_a_df = freq_dist[freq_dist[self.label_name] == classes[0]]
        class_b_df = freq_dist[freq_dist[self.label_name] == classes[1]]

        on = ["amino acid", "position"]
        on = on + ["locus"] if "locus" in freq_dist.columns else on

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
                        facet_row="locus" if "locus" in frequency_change.columns else None,
                        color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                        labels={"position": "IMGT position" if self.alignment == "IMGT" else "Position",
                                "positive_class": self.label_name,
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
                    logging.warning(f"{AminoAcidFrequencyDistribution.__name__}: ambiguous label: split_by_label was "
                                    f"set to True but no label name was specified, and the number of available labels "
                                    f"is {len(self.dataset.get_label_names())}: {self.dataset.get_label_names()}. "
                                    f"Skipping this report...")
                    return False
            elif self.label_name not in self.dataset.get_label_names():
                logging.warning(f"{AminoAcidFrequencyDistribution.__name__}: the specified label name "
                                f"({self.label_name}) was not available among the dataset labels: "
                                f"{self.dataset.get_label_names()}. Skipping this report...")
                return False

        return True
