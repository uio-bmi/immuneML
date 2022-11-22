import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.PositionHelper import PositionHelper


class AminoAcidFrequencyDistribution(DataReport):
    """
    Generates a barplot showing the relative frequency of each amino acid at each position in the sequences of a dataset.

    Arguments:

        imgt_positions (bool): Whether to use IMGT positional numbering or sequence index numbering. When imgt_positions is True, IMGT positions are used, meaning sequences of unequal length are aligned according to their IMGT positions. By default imgt_positions is True.

        relative_frequency (bool): Whether to plot relative frequencies (true) or absolute counts (false) of the positional amino acids. By default, relative_frequency is True.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_aa_freq_report: AminoAcidFrequencyDistribution

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = AminoAcidFrequencyDistribution.__name__
        ParameterValidator.assert_type_and_value(kwargs["imgt_positions"], bool, location, "imgt_positions")
        ParameterValidator.assert_type_and_value(kwargs["relative_frequency"], bool, location, "relative_frequency")

        return AminoAcidFrequencyDistribution(**kwargs)

    def __init__(self, dataset: SequenceDataset = None, imgt_positions: bool = None, relative_frequency: bool = None,
                 result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.imgt_positions = imgt_positions
        self.relative_frequency = relative_frequency
    #
    # def check_prerequisites(self):
    #     return True
        # if isinstance(self.dataset, SequenceDataset):
        #     return True
        # else:
        #     warnings.warn(f"{self.__class__.__name__}: report can be generated only from SequenceDataset. Skipping this report...")
        #     return False

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        freq_dist = self._get_plotting_data()
        results_table = self._write_results_table(freq_dist)
        report_output_fig = self._safe_plot(freq_dist=freq_dist)

        return ReportResult(name=self.name,
                            info="A a barplot showing the relative frequency of each amino acid at each position in the sequences of a dataset.",
                            output_figures=None if report_output_fig is None else [report_output_fig],
                            output_tables=None if results_table is None else [results_table])

    def _get_plotting_data(self):
        if isinstance(self.dataset, SequenceDataset):
            return self._count_dict_to_df(self._count_aa_frequencies(self.dataset.get_data()))
        elif isinstance(self.dataset, ReceptorDataset):
            result_dfs = []

            receptors = self.dataset.get_data()
            chains = next(receptors).get_chains()

            for chain in chains:
                result_df = self._count_dict_to_df(self._count_aa_frequencies(self._chain_iterator(chain)))
                result_df["chain"] = chain
                result_dfs.append(result_df)

            return pd.concat(result_dfs)
        elif isinstance(self.dataset, RepertoireDataset):
            raw_count_dict = {}

            for repertoire in self.dataset.get_data():
                self._count_aa_frequencies(repertoire.get_sequence_objects(), raw_count_dict)

            return self._count_dict_to_df(raw_count_dict)

    def _chain_iterator(self, chain):
        for receptor in self.dataset.get_data():
            assert chain in receptor.get_chains(), f"{AminoAcidFrequencyDistribution.__name__}: All receptors in the dataset must contain the same chains. Expected {chain} but found {receptor.get_chains()}"
            yield receptor.get_chain(chain)

    def _count_aa_frequencies(self, sequence_iterator, raw_count_dict=None):
        raw_count_dict = {} if raw_count_dict is None else raw_count_dict

        for sequence in sequence_iterator:
            seq_str = sequence.get_sequence(sequence_type=SequenceType.AMINO_ACID)
            seq_pos = self._get_positions(sequence)

            for aa, pos in zip(seq_str, seq_pos):
                if pos not in raw_count_dict:
                    raw_count_dict[pos] = {legal_aa: 0 for legal_aa in
                                           EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID)}

                raw_count_dict[pos][aa] += 1

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
            positions = PositionHelper.gen_imgt_positions_from_length(len(sequence.get_sequence(SequenceType.AMINO_ACID)),
                                                                      sequence.get_attribute("region_type"))
        else:
            positions = list(range(len(sequence.get_sequence(SequenceType.AMINO_ACID))))

        return [str(pos) for pos in positions]

    def _write_results_table(self, results_table):
        file_path = self.result_path / "amino_acid_frequency_distribution.csv"

        results_table.to_csv(file_path, index=False)

        return ReportOutput(path=file_path, name="Table of amino acid frequencies")

    def _plot(self, freq_dist):
        y = "relative frequency" if self.relative_frequency else "count"

        figure = px.bar(freq_dist, x="position", y=y, color="amino acid", text="amino acid",
                        facet_col="chain" if "chain" in freq_dist.columns else None,
                        labels={"position": "IMGT position" if self.imgt_positions else "Sequence index",
                                "count": "Count",
                                "relative frequency": "Relative frequency",
                                "amino acid": "Amino acid"})
        figure.update_xaxes(categoryorder='array', categoryarray=self._get_position_order(freq_dist["position"]))
        figure.update_layout(showlegend=False)

        if self.relative_frequency:
            figure.update_layout(yaxis={"tickformat": ",.0%", "range": [0,1]})

        file_path = self.result_path / "amino_acid_frequency_distribution.html"
        figure.write_html(str(file_path))

        return ReportOutput(path=file_path, name="Amino acid frequency distribution")

    def _get_position_order(self, positions):
        return [str(int(pos)) if pos.is_integer() else str(pos) for pos in sorted(set(positions.astype(float)))]



