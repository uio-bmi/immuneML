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
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceGenerationProbabilityDistribution(DataReport):
    """
    Generates a plot of the distribution of generation probability and appearance rate of the sequences in a RepertoireDataset.

    Arguments:
        count_by_repertoire (bool): Whether to have the appearance rate of a sequence be decided by how many repertoires it
        appears in (True) or by the total sequence count in all repertoires (False).
        Default value is False.
        mark_implanted_labels (bool): Plot the implanted sequences with different colors. Default value is True.
        default_sequence_label (str): Name of main dataset (all non-implanted sequences) in the plot legend. Default value is "dataset".

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_p_gen_report:
            SequenceGenerationProbabilityDistribution:
                count_by_repertoire: False
                mark_implanted_labels: True
                default_sequence_label: OLGA
    """

    # TODO change code so that a dataframe is passed down instead of many dicts
    # TODO add different colors based on repertoire metadata (sick/healthy)

    @classmethod
    def build_object(cls,
                     **kwargs):  # called when parsing YAML - all checks for parameters (if any) should be in this function
        return SequenceGenerationProbabilityDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None,
                 name: str = None, count_by_repertoire: bool = False, mark_implanted_labels: bool = True,
                 default_sequence_label: str = "dataset"):
        super().__init__(dataset=dataset, result_path=result_path, name=name)

        #ParameterValidator.

        self.count_by_repertoire = count_by_repertoire
        self.mark_implanted_labels = mark_implanted_labels
        self.default_sequence_label = default_sequence_label

    def check_prerequisites(
            self):  # called at runtime to check if the report can be run with params assigned at runtime (e.g., dataset is set at runtime)
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            logging.warning("Report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:  # the function that creates the report

        if self.mark_implanted_labels:
            sequence_labels = self._get_sequence_labels()
        else:
            sequence_labels = {}
        sequence_count = self._get_sequence_count()
        sequence_pgen = self._get_sequence_pgen()

        report_output_fig = self._safe_plot(sequence_count=sequence_count, sequence_pgen=sequence_pgen,
                                            sequence_labels=sequence_labels)
        output_figures = None if report_output_fig is None else [report_output_fig]

        # TODO dataframe as output table (df.to_csv(..., sep="\t"))
        output_tables = None

        return ReportResult(type(self).__name__, output_figures=output_figures, output_tables=output_tables)

    def _get_sequence_labels(self):
        """
        Retrieves sequence labels from self.dataset. Label is taken from signal_id label from implanted
        sequences. Sequences without ImplantAnnotation are assigned default sequence label

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is sequence label (str)
        """

        label_names = list(self.dataset.get_label_names())
        sequence_labels = {}

        for repertoire in self.dataset.get_data():
            rep_attributes = repertoire.get_attributes(["sequence_aas"] + label_names)

            for i in range(len(rep_attributes["sequence_aas"])):
                # TODO account for the fact that signal can show up naturally and implanted
                if rep_attributes["sequence_aas"][i] in sequence_labels and sequence_labels[rep_attributes["sequence_aas"][i]] != "natural":
                    continue

                seq_label = None

                for label in label_names:
                    if label in rep_attributes:
                        if rep_attributes[label][i]:
                            seq_label = re.findall("signal_id='([^\"']+)", rep_attributes[label][i])[0]
                            break

                if not seq_label:
                    seq_label = self.default_sequence_label

                sequence_labels[rep_attributes["sequence_aas"][i]] = seq_label

        return sequence_labels

    def _get_sequence_pgen(self):
        """
        Computes generation probability of each sequence from self.dataset.

        Returns:
            np.ndarray: Generation probabilities of sequences
        """

        # TODO get default_model_name from dataset (organism + chain type) (organism is not that important, focus on chain type)
        # check if set(get_attribute("chains")) contains only one chain type
        # at least support human TRB, TRA, IGH, IGL, IGK. If the chain type is unknown, raise an error
        olga = OLGA.build_object(model_path=None, default_model_name="humanTRB", chain=None, use_only_productive=False)
        olga.load_model()

        df = pd.DataFrame()

        for repertoire in self.dataset.get_data():
            attr = pd.DataFrame(data=repertoire.get_attributes(["sequence_aas", "v_genes", "j_genes"]))
            df = pd.concat([df, attr], ignore_index=True)

        df.drop_duplicates(subset=["sequence_aas"], inplace=True)

        return olga.compute_p_gens(df, SequenceType.AMINO_ACID)

    def _get_sequence_count(self):
        """
        Either counts number of duplicates of each sequence from dataset, or how many repertoires each sequence
        appears in. This is specified in yaml file.

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is sequence count (int)
        """
        sequence_count = {}

        if self.count_by_repertoire:
            counting_func = self._get_repertoire_appearance_rate
        else:
            counting_func = self._get_total_sequence_count

        for repertoire in self.dataset.get_data():
            print(f"Repertoire size: {len(repertoire.get_sequence_aas())}")
            for seq, count in counting_func(repertoire).items():
                sequence_count[seq] = sequence_count[seq] + count if seq in sequence_count else count

        return sequence_count

    @staticmethod
    def _get_total_sequence_count(repertoire: Repertoire) -> dict:
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

    @staticmethod
    def _get_repertoire_appearance_rate(repertoire: Repertoire) -> dict:
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

    def _plot(self, sequence_count, sequence_pgen, sequence_labels) -> ReportOutput:

        if self.mark_implanted_labels:
            df = pd.DataFrame({
                "pgen": sequence_pgen,
                "sequence_count": list(sequence_count.values()),
                "sequence": list(sequence_count.keys()),
                "sequence_labels": list(sequence_labels.values())
            })

            # jitter
            figure = px.strip(df, x="sequence_count", y="pgen", hover_data=["sequence"], color="sequence_labels", stripmode="overlay")
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

        figure.update_traces(jitter=1.0)

        PathBuilder.build(self.result_path)

        file_path = self.result_path / "pgen_scatter_plot.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="sequence generation probability distribution plot")
