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
from immuneML.util.ParameterValidator import ParameterValidator
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


    Specification arguments:

        motif_color_map (dict): An optional mapping between motif sizes and colors. If no mapping is given, default colors will be chosen.


    YAML specification example:

    .. indent with spaces
    .. code-block:: yaml

        my_motif_sim:
            NonMotifSimilarity:
                motif_color_map:
                    3: "#66C5CC"
                    4: "#F6CF71"
                    5: "#F89C74"

    """

    @classmethod
    def build_object(cls, **kwargs):
        if "motif_color_map" in kwargs:
            ParameterValidator.assert_type_and_value(kwargs["motif_color_map"], dict, NonMotifSequenceSimilarity.__name__, "motif_color_map")
            kwargs["motif_color_map"] = {str(key): value for key, value in kwargs["motif_color_map"].items()}

        return NonMotifSequenceSimilarity(**kwargs)

    def __init__(self, dataset: SequenceDataset = None, result_path: Path = None,
                 motif_color_map: dict = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.sequence_length = 0
        self.motif_color_map = motif_color_map

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
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(self.dataset)
        self.sequence_length = len(np_sequences[0])

        raw_counts = pd.DataFrame([self._make_hamming_distance_hist_for_motif(motif_presence, np_sequences)
                                   for motif_presence in self.dataset.encoded_data.examples.T])

        ### Original code with multiprocessing (fails with bionumpy + pickle error?)
        # with Pool(processes=self.number_of_processes) as pool:
        #     partial_func = partial(self._make_hamming_distance_hist_for_motif,  np_sequences=np_sequences)
        #     raw_counts = pd.DataFrame(pool.map(partial_func, self.dataset.encoded_data.examples.T))

        raw_counts["motif"] = self.dataset.encoded_data.feature_names

        return raw_counts
    def _make_hamming_distance_hist_for_motif(self, motif_presence, np_sequences):
        positive_sequences = np_sequences[motif_presence]
        return self._make_hamming_distance_hist(positive_sequences)
    def _make_hamming_distance_hist(self, sequences):
        counts = {i: 0 for i in range(self.sequence_length)}
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
        if self.motif_color_map is not None:
            color_discrete_map = self.motif_color_map
            color_discrete_sequence = None
        else:
            color_discrete_map = None
            color_discrete_sequence = px.colors.sequential.Sunsetdark

        plotting_data["motif_size"] = plotting_data["motif_size"].astype(str)

        fig = px.line(plotting_data, x='Hamming', y='Percentage', markers=True, line_group='motif', color="motif_size",
                      template='plotly_white', color_discrete_sequence=color_discrete_sequence, color_discrete_map=color_discrete_map,
                      labels={"Percentage": "Percentage of sequence pairs containing motif",
                              "Hamming": "Hamming distance between sequences",
                              "motif_size": "Motif length<br>(number of<br>amino acids)"})

        fig.layout.yaxis.tickformat = ',.0%'

        fig.update_traces(opacity=0.7)

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
