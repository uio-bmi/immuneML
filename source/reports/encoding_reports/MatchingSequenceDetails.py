import csv

from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
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

    def generate(self, dataset: Dataset, result_path: str, params: dict):

        PathBuilder.build(result_path)
        self._make_overview(dataset, result_path, params)
        self._make_matching_report(dataset, result_path, params)

    def _make_overview(self, dataset: Dataset, result_path: str, params: dict):
        filename = result_path + "matching_sequence_overview.tsv"
        fieldnames = ["patient", "chain", dataset.encoded_data["feature_names"][0],
                      "repertoire_size", "max_levenshtein_distance"]
        for label in dataset.params.keys():
            fieldnames.append("{}_status".format(label))
        self._write_rows(dataset, params, filename, fieldnames)

        return filename

    def _write_rows(self, dataset: Dataset, params: dict, filename: str, fieldnames: list):
        with open(filename, "w") as file:
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
            csv_writer.writeheader()
            for index, repertoire in enumerate(dataset.get_data()):
                row = {
                    "patient": repertoire.identifier,
                    "chain": str(list({seq.metadata.chain for seq in repertoire.sequences}))[1:-1],
                    dataset.encoded_data["feature_names"][0]: dataset.encoded_data["repertoires"][index][0],
                    "repertoire_size": len(repertoire.sequences),
                    "max_levenshtein_distance": params["max_distance"]
                }
                for label in dataset.params.keys():
                    row["{}_status".format(label)] = repertoire.metadata.custom_params[label]
                csv_writer.writerow(row)

    def _make_matching_report(self, dataset: Dataset, result_path: str, params: dict):

        filenames = []

        for repertoire in dataset.get_data():
            filenames.append(self._make_repertoire_report(repertoire, result_path, params))

        return filenames

    def _make_repertoire_report(self, repertoire: Repertoire, result_path: str, params: dict):

        filename = result_path + "{}_{}.tsv".format(repertoire.identifier,
                                                    list({seq.metadata.chain for seq in repertoire.sequences}))

        with open(filename, "w") as file:
            csv_writer = csv.DictWriter(file, fieldnames=["sequence", "v_gene", "j_gene", "chain", "matching_sequences", "max_distance"], delimiter="\t")
            csv_writer.writeheader()
            for index, sequence in enumerate(repertoire.sequences):

                matching_sequences = self._find_matching_sequences(sequence, params["reference_sequences"], params["max_distance"])

                csv_writer.writerow({
                    "sequence": sequence.get_sequence(),
                    "v_gene": sequence.metadata.v_gene,
                    "j_gene": sequence.metadata.j_gene,
                    "chain": sequence.metadata.chain,
                    "matching_sequences": str(matching_sequences)[1:-1].replace("'", ""),
                    "max_distance": params["max_distance"]
                })

        return filename

    def _find_matching_sequences(self, sequence: ReceptorSequence, reference_sequences: list, max_distance: int):
        matcher = SequenceMatcher()
        return matcher.match_sequence(sequence, reference_sequences, max_distance)["matching_sequences"]
