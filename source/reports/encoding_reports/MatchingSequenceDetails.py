import csv
import warnings

from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.analysis.SequenceMatcher import SequenceMatcher
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.repertoire.Repertoire import Repertoire
from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.ImportHelper import ImportHelper
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder
from source.util.ReflectionHandler import ReflectionHandler


class MatchingSequenceDetails(EncodingReport):
    """
    Compares the sequences in the repertoires of a given RepertoireDataset to a list of reference sequences,
    and reports the number of matching sequences.

    The resulting matching_sequence_overview.tsv contains:
        - repertoire_identifier: the unique identifier of the repertoire
        - count/percentage/clonal_percentage: # todo fill this in
        - repertoire_size: the total number of different sequences (i.e. clonotypes) in the repertoire
        - max_levenshtein_distance: the maximum lehvenstein distance that was used to calculate the matches
        - Any repertoire labels and their values

    # todo refactor this class: either merge with MatchedPaired classes or remove redundant information from this report
    # (max edit distance and reference sequences are also present in the encoding)


    # todo old stuff, remove
    params:
        - list of reference sequences
        - max Levenshtein distance
        - summary:  * count the number of sequences from the repertoire matched,
                    * get the percentage of sequences from the repertoire matched,
                    * get the percentage of sequences from the repertoire matched with respect to clonal counts

    """

    @classmethod
    def build_object(cls, **kwargs):

        location = "MatchingSequenceDetails"

        if "max_edit_distance" in kwargs:
            ParameterValidator.assert_type_and_value(kwargs["max_edit_distance"], int, location, "max_edit_distance")

        if "reference_sequences" in kwargs:
            ParameterValidator.assert_keys(list(kwargs["reference_sequences"].keys()), ["format", "path"], location, "reference_sequences")

            format_str = kwargs["reference_sequences"]["format"]

            import_class = ReflectionHandler.get_class_by_name("{}Import".format(format_str))
            params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path + "datasets/", DefaultParamsLoader._convert_to_snake_case(format_str))
            params["paired"] = False
            params["is_repertoire"] = False
            processed_params = DatasetImportParams.build_object(**params)

            sequences = ImportHelper.import_items(import_class, kwargs["reference_sequences"]["path"], processed_params)
            kwargs["reference_sequences"] = sequences

        return MatchingSequenceDetails(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, max_edit_distance: int = None, reference_sequences: ReceptorSequenceList = None,
                 result_path: str = None, name: str = None):

        self.dataset = dataset
        self.max_edit_distance = max_edit_distance
        self.reference_sequences = reference_sequences
        self.result_path = result_path
        self.name = name

    def generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        overview_file = self._make_overview()
        filenames = self._make_matching_report()
        return ReportResult(self.name, output_tables=[ReportOutput(item) for item in [overview_file] + filenames])

    def check_prerequisites(self):
        if "MatchedReferenceEncoder" != self.dataset.encoded_data.encoding:
            warnings.warn("Encoding is not compatible with the report type. MatchingSequenceDetails report will not be created.")
            return False
        else:
            return True

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
                    "max_levenshtein_distance": self.max_edit_distance
                }
                for label in self.dataset.params.keys():
                    row["{}".format(label)] = repertoire.metadata[label]
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
                    "max_distance": self.max_edit_distance
                })

        return filename

    def _find_matching_sequences(self, sequence: ReceptorSequence):
        matcher = SequenceMatcher()
        return matcher.match_sequence(sequence, self.reference_sequences, self.max_edit_distance)["matching_sequences"]
