from pathlib import Path

import logging
import plotly.express as px
import pandas as pd
from multiprocessing import Pool
from functools import partial


from immuneML.data_model.dataset import SequenceDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class NonMotifSequenceSimilarity(EncodingReport):
    """
    Plots the similarity of positions outside the motifs of interest. This report can be used to investigate if the
    motifs of interest as determined by the :py:obj:`~immuneML.encodings.motif_encoding.MotifEncoder.MotifEncoder`
    have a tendency occur in sequences that are naturally very similar or dissimilar.

    For each motif, the subset of sequences containing the motif is selected, and the hamming distances are computed
    between all sequences in this subset. Finally, a plot is created showing the distribution of hamming distances
    between the sequences containing the motif. For motifs occurring in sets of very similar sequences, this distribution
    will lean towards small hamming distances. Likewise, for motifs occurring in a very diverse set of sequences, the
    distribution will lean towards containing more large hamming distances.

    YAML specification example:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_sim: NonMotifSimilarity

    """

    @classmethod
    def build_object(cls, **kwargs):
        return NonMotifSequenceSimilarity(**kwargs)

    def __init__(self, dataset: SequenceDataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)

    def _generate(self):
        PathBuilder.build(self.result_path)

        raw_counts = self.get_hamming_distance_counts()
        plotting_data = self.get_plotting_data(raw_counts)

        table_1 = self._write_output_table(raw_counts, self.result_path / "sequence_hamming_distances_raw.tsv",
                                                "Hamming distances between sequences sharing the same motif, raw counts.")
        table_2 = self._write_output_table(plotting_data, self.result_path / "sequence_hamming_distances_percentage.tsv",
                                                "Hamming distances between sequences sharing the same motif, expressed as percentage.")

        output_figure = self._safe_plot(plotting_data=plotting_data)

        return ReportResult(
            name=self.name,
            output_figures=[output_figure],
            output_tables=[table_1, table_2],
        )

    def get_hamming_distance_counts(self):
        # np_sequences, y_true, _ = read_data_file(data_file=data_file, calculate_weights=False)
        # motifs = filter_single_motifs(read_motifs_from_file(motifs_file))

        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(self.dataset)

        with Pool(processes=self.number_of_processes) as pool:
            partial_func = partial(self._make_hamming_distance_hist_for_motif,  np_sequences=np_sequences)
            raw_counts = pd.DataFrame(pool.map(partial_func, self.dataset.encoded_data.examples.T))

        # df = pd.DataFrame(hd_counts)
        raw_counts["motif"] = self.dataset.encoded_data.feature_names

        return raw_counts
    def _make_hamming_distance_hist_for_motif(self, motif_presence, np_sequences):
        positive_sequences = np_sequences[motif_presence]
        return self._make_hamming_distance_hist(positive_sequences)
    def _make_hamming_distance_hist(self, sequences):
        counts = {i: 0 for i in range(len(sequences[0]))}
        for dist in self._calculate_all_hamming_distances(sequences):
            counts[dist] += 1

        return counts
    def _calculate_all_hamming_distances(self, sequences):
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                yield sum(sequences[i] != sequences[j])

    def get_plotting_data(self, raw_counts):
        motif_col = raw_counts.loc[:,"motif"]
        counts = raw_counts.loc[:, raw_counts.columns != "motif"]

        total = counts.apply(sum, axis=1)
        plotting_data = counts.div(total, axis=0)
        plotting_data["motif"] = motif_col
        plotting_data = plotting_data[total > 0]
        plotting_data = plotting_data.loc[::-1]
        plotting_data = pd.melt(plotting_data, id_vars=["motif"], var_name="Hamming", value_name="Percentage")

        plotting_data["Hamming"] = plotting_data["Hamming"].astype(str)
        plotting_data["motif_size"] = plotting_data["motif"].apply(lambda motif: len(motif.split("-")[0].split("&")))

        return plotting_data

    def _plot(self, plotting_data) -> ReportOutput:
        fig = px.line(plotting_data, x='Hamming', y='Percentage', markers=True, line_group='motif', color="motif_size",
                      template='plotly_white', color_discrete_sequence=px.colors.sequential.Sunsetdark,
                      labels={"Percentage": "Percentage of sequence pairs sharing the motif",
                              "Hamming": "Hamming distance between pairs of sequences sharing the motif",
                              "motif_size": "Motif length<br>(number of<br>amino acids)"})
        fig.layout.yaxis.tickformat = ',.0%'

        fig.update_layout(
            font=dict(
                size=14,
            )
        )

        output_path = self.result_path / "sequence_hamming_distances.html"

        fig.write_html(str(output_path))

        return ReportOutput(path=output_path, name="Hamming distances between sequences sharing the same motif")

    def check_prerequisites(self):
        valid_encodings = [MotifEncoder.__name__]

        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning(f"{self.__class__.__name__}: the dataset is not encoded, skipping this report...")
            return False
        elif self.dataset.encoded_data.encoding not in valid_encodings:
            logging.warning(f"{self.__class__.__name__}: the dataset encoding ({self.dataset.encoded_data.encoding}) was not in the list of valid "
                            f"encodings ({valid_encodings}), skipping this report...")
            return False
        else:
            return True
