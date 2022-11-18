import logging
import re
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.util.PathBuilder import PathBuilder


class SequenceGenerationProbabilityDistribution(DataReport):
    """
    Generates a plot of the distribution of generation probability and appearance rate of the sequences in a RepertoireDataset.

    Arguments:
        count_by_repertoire (bool): Whether to have the appearance rate of a sequence be decided by how many repertoires it
        appears in (True) or by the total sequence count in all repertoires (False).
        Default value is False.
        mark_implanted_labels (bool): Plot the implanted sequences with different colors. Default value is True.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_p_gen_report:
            SequenceGenerationProbabilityDistribution:
                count_by_repertoire: False
                mark_implanted_labels: True
    """

    @classmethod
    def build_object(cls,
                     **kwargs):  # called when parsing YAML - all checks for parameters (if any) should be in this function
        # TODO check if argument is legal
        return SequenceGenerationProbabilityDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, batch_size: int = 1, result_path: Path = None,
                 name: str = None, count_by_repertoire: bool = False, mark_implanted_labels: bool = True):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.batch_size = batch_size
        self.count_by_repertoire = count_by_repertoire
        self.mark_implanted_labels = mark_implanted_labels

    def check_prerequisites(
            self):  # called at runtime to check if the report can be run with params assigned at runtime (e.g., dataset is set at runtime)
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            logging.warning("Report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:  # the function that creates the report

        if self.mark_implanted_labels:
            print("Start generation method", "\n")
            generation_method = self._get_generation_method()
        else:
            generation_method = {}
        print("Start count", "\n")
        sequence_count = self._get_sequence_count()
        print("Start pgen computation", "\n")
        sequence_pgen = self._get_sequence_pgen()

        report_output_fig = self._safe_plot(sequence_count=sequence_count, sequence_pgen=sequence_pgen,
                                            generation_method=generation_method)
        output_figures = None if report_output_fig is None else [report_output_fig]

        return ReportResult(type(self).__name__, output_figures=output_figures)

    def _get_generation_method(self):
        """
        Retrieves method of generation from self.dataset. Generation method is taken from signal_id label from implanted
        sequences. Sequences without ImplantAnnotation are assigned value "natural"

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is generation method (str)
        """

        label_names = list(self.dataset.get_label_names())
        generation_method = {}

        '''
        for repertoire, meta_dataset in zip(self.dataset.get_data(self.batch_size),
                                            self.dataset.get_metadata(["dataset"])["dataset"]):
            rep_attributes = repertoire.get_attributes(["sequence_aas"] + label_names)

            for i in range(len(rep_attributes["sequence_aas"])):
                generation_method[rep_attributes["sequence_aas"][i]] = meta_dataset
        '''

        for repertoire in self.dataset.get_data(self.batch_size):
            rep_attributes = repertoire.get_attributes(["sequence_aas"] + label_names)

            for i in range(len(rep_attributes["sequence_aas"])):
                # TODO account for the fact that signal can show up naturally and implanted
                if rep_attributes["sequence_aas"][i] in generation_method and generation_method[rep_attributes["sequence_aas"][i]] != "natural":
                    continue

                seq_gen_method = None

                for label in label_names:
                    if label in rep_attributes:
                        if rep_attributes[label][i]:
                            # TODO do ImplantAnnotation parsing better
                            seq_gen_method = re.split("=|'|,|\s", rep_attributes[label][i])[2]
                            break

                if not seq_gen_method:
                    seq_gen_method = "natural"

                generation_method[rep_attributes["sequence_aas"][i]] = seq_gen_method

        return generation_method

    def _get_sequence_pgen(self):
        """
        Computes generation probability of each sequence from self.dataset.

        Returns:
            np.ndarray: Generation probabilities of sequences
        """

        olga = OLGA.build_object(model_path=None, default_model_name="humanTRB", chain=None, use_only_productive=False)
        olga.load_model()

        df = pd.DataFrame()

        for repertoire in self.dataset.get_data(self.batch_size):
            attr = pd.DataFrame(data=repertoire.get_attributes(["sequence_aas", "v_genes", "j_genes"]))
            df = pd.concat([df, attr], ignore_index=True)

        df.drop_duplicates(subset=["sequence_aas"], inplace=True)

        return olga.compute_p_gens(df, SequenceType.AMINO_ACID)

    def _get_sequence_count(self):
        """
        Either counts number of duplicates of each sequence from self.dataset, or how many repertoires each sequence
        appears in. This is specified in yaml file.

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is sequence count (int)
        """
        sequence_count = {}

        if self.count_by_repertoire:
            counting_func = self._get_repertoire_appearance_rate
        else:
            counting_func = self._get_total_sequence_count

        for repertoire in self.dataset.get_data(self.batch_size):
            print(f"Repertoire size: {len(repertoire.get_sequence_aas())}")
            for seq, count in counting_func(repertoire).items():
                sequence_count[seq] = sequence_count[seq] + count if seq in sequence_count else count

        return sequence_count

    def _get_total_sequence_count(self, repertoire: Repertoire) -> dict:
        """
        Counts number of duplicates of each sequence in repertoire.

        Args:
            repertoire (Repertoire): Repertoire

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is sequence count (int)
        """
        sequence_count = {}

        for seq, count in zip(repertoire.get_sequence_aas(), repertoire.get_counts()):
            sequence_count[seq] = sequence_count[seq] + count if seq in sequence_count else count

        return sequence_count

    def _get_repertoire_appearance_rate(self, repertoire: Repertoire) -> dict:
        """
        Marks sequences that appear in repertoire with 1.

        Args:
            repertoire (Repertoire): Repertoire

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is 1.
        """
        sequence_count = {}

        for seq in repertoire.get_sequence_aas():
            sequence_count[seq] = 1

        return sequence_count

    def _plot(self, sequence_count, sequence_pgen, generation_method) -> ReportOutput:

        if self.mark_implanted_labels:
            df = pd.DataFrame({
                "pgen": sequence_pgen,
                "sequence_count": list(sequence_count.values()),
                "sequence": list(sequence_count.keys()),
                "generation_method": list(generation_method.values())
            })

            # jitter
            figure = px.strip(df, x="sequence_count", y="pgen", hover_data=["sequence"], color="generation_method", stripmode="overlay")
        else:
            df = pd.DataFrame({
                "pgen": sequence_pgen,
                "sequence_count": list(sequence_count.values()),
                "sequence": list(sequence_count.keys())
            })

            # jitter
            figure = px.strip(df, x="sequence_count", y="pgen", hover_data=["sequence"])

        print(f"Number of single seqs: {len(df[df['sequence_count'] == 1])} / {len(df)} = {len(df[df['sequence_count'] == 1]) / len(df)}")

        xaxis_title = "nr. of repertoires the sequence appears in" if self.count_by_repertoire else "total sequence count"

        figure.update_layout(title="Sequence generation probability distribution", template="plotly_white",
                             xaxis=dict(tickmode='array', tickvals=list(range(1, max(df["sequence_count"]) + 1))),
                             yaxis=dict(showexponent='all', exponentformat='e', type="log"),
                             xaxis_title=xaxis_title)
        # range_y=[float(df[df["pgen"] > 0]["pgen"].min()) * 0.5, float(df['pgen'].max()) * 1.5]
        figure.update_traces(jitter=1.0)

        PathBuilder.build(self.result_path)

        # export AIRR dataset
        AIRRExporter.export(self.dataset, self.result_path)

        file_path = self.result_path / "pgen_scatter_plot.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="sequence generation probability distribution plot")
