import csv

from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class MatchingSequenceDetails(EncodingReport):
    """
    TODO: write description here

    params:
        - list of reference sequences
        - max Levenshtein distance
        - summary:  * count the number of sequences from the repertoire matched,
                    * get the percentage of sequences from the repertoire matched,
                    * get the percentage of sequences from the repertoire matched with respect to clonal counts
    """

    def __init__(self, dataset: RepertoireDataset = None, max_distance: int = None, reference_sequences: list = None, result_path: str = None):
        self.dataset = dataset
        self.max_distance = max_distance
        self.reference_sequences = reference_sequences
        self.result_path = result_path

    def generate(self):
        PathBuilder.build(self.result_path)
        self._make_overview()
        self._make_matching_report()

    def check_prerequisites(self):
        assert "MatchedReferenceEncoder" == self.dataset.encoded_data.encoding, "Encoding is not compatible with the report type. " \
                                                                                "MatchingSequenceDetails report will not be created."

    def _make_overview(self):
        filename = self.result_path + "matching_sequence_overview.tsv"
        fieldnames = ["repertoire_identifier", self.dataset.encoded_data.feature_names[0],
                      "repertoire_size", "max_levenshtein_distance"]
        for label in self.dataset.params.keys():
            fieldnames.append("{}".format(label))
        self._write_rows(filename, fieldnames)

        return filename

    def _write_rows(self, filename: str, fieldnames: list):
        with open(filename, "w") as file:
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
            csv_writer.writeheader()
            for index, repertoire in enumerate(self.dataset.get_data()):
                row = {
                    "repertoire_identifier": repertoire.identifier,
                    self.dataset.encoded_data.feature_names[0]: self.dataset.encoded_data.examples[index][0],
                    "repertoire_size": len(repertoire.sequences),
                    "max_levenshtein_distance": self.max_distance
                }
                for label in self.dataset.params.keys():
                    row["{}".format(label)] = repertoire.metadata.custom_params[label]
                csv_writer.writerow(row)

    def _make_matching_report(self):

        filenames = []

        for repertoire in self.dataset.get_data():
            filenames.append(self._make_repertoire_report(repertoire))

        return filenames

    def _make_repertoire_report(self, repertoire: Repertoire):

        filename = self.result_path + "{}_{}.tsv".format(repertoire.identifier,
                                                         list({seq.metadata.chain for seq in repertoire.sequences}))

        with open(filename, "w") as file:
            csv_writer = csv.DictWriter(file, fieldnames=["sequence", "v_gene", "j_gene", "chain", "clone_count", "matching_sequences", "max_distance"], delimiter="\t")
            csv_writer.writeheader()
            for index, sequence in enumerate(repertoire.sequences):

                matching_sequences = self._find_matching_sequences(sequence)

                csv_writer.writerow({
                    "sequence": sequence.get_sequence(),
                    "v_gene": sequence.metadata.v_gene,
                    "j_gene": sequence.metadata.j_gene,
                    "chain": sequence.metadata.chain,
                    "clone_count": sequence.metadata.count,
                    "matching_sequences": str(matching_sequences)[1:-1].replace("'", ""),
                    "max_distance": self.max_distance
                })

        return filename

    def _find_matching_sequences(self, sequence: ReceptorSequence):
        matcher = SequenceMatcher()
        return matcher.match_sequence(sequence, self.reference_sequences, self.max_distance)["matching_sequences"]
