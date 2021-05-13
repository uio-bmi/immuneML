from pathlib import Path
import subprocess
from io import StringIO
import pandas as pd
import warnings
from tempfile import NamedTemporaryFile


from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.DistanceMetricType import DistanceMetricType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from scripts.specification_util import update_docs_per_mapping


class EditDistanceEncoder(DatasetEncoder):
    """
    Encodes a given RepertoireDataset as a distance matrix, internally using `MatchAIRR <https://github.com/uio-bmi/vdjsearch/>`_
    for fast computation. This creates a pairwise distance matrix between each of the repertoires.
    The distance is calculated based on the number of matching receptor chain sequences between the repertoires. This matching may be
    defined to permit 1 or 2 mismatching amino acid/nucleotide positions and 1 indel in the sequence. Furthermore,
    matching may or may not include V and J gene information, and sequence frequencies may be included or ignored.


    Arguments:

        matchairr_path (Path): path to the MatchAIRR executable

        distance_metric (str): The distance metric to be applied after computing the number of overlapping sequences.
        The value is ignored, Jaccard is computed for now.

        differences (int): Number of differences allowed between the sequences of two immune receptor chains, this
        may be between 0 and 2. By default differences is 0.

        indels (bool): Whether to allow an indel. This is only possible if differences is 1. By default, indels is False.

        ignore_frequency (bool): Whether to ignore the frequencies of the immune receptor chains. If False, frequencies
        will be included, meaning the 'counts' values for the receptors available in two repertoires are multiplied.
        If False, only the number of unique overlapping immune receptors ('clones') are considered.
        By default, ignore_frequency is False.

        ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains
        have to match. If True, gene information is ignored. By default, ignore_genes is False.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_distance_encoder:
            Distance:
                matchairr_path: path/to/matchairr
                distance_metric: JACCARD
                # Optional parameters:
                differences: 0
                indels: False
                ignore_frequency: False
                ignore_genes: False

    """

    def __init__(self, matchairr_path: Path, distance_metric: DistanceMetricType, differences: int, indels: bool, ignore_frequency: bool, ignore_genes: bool, context: dict = None, name: str = None):
        self.matchairr_path = Path(matchairr_path)
        self.distance_metric = distance_metric
        self.differences = differences
        self.indels = indels
        self.ignore_frequency = ignore_frequency
        self.ignore_genes = ignore_genes
        self.context = context
        self.name = name
        self.comparison = None

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def _prepare_parameters(matchairr_path: str, distance_metric: str, differences: int, indels: bool, ignore_frequency: bool, ignore_genes: bool, context: dict = None, name: str = None):
        #todo supply other distance metrics
        ParameterValidator.assert_type_and_value(differences, int, "EditDistanceEncoder", "differences", min_inclusive=0, max_inclusive=2)
        ParameterValidator.assert_type_and_value(indels, bool, "EditDistanceEncoder", "indels")
        if indels:
            assert differences == 1, f"EditDistanceEncoder: If indels is True, differences is only allowed to be 1, found {differences}"

        ParameterValidator.assert_type_and_value(ignore_frequency, bool, "EditDistanceEncoder", "ignore_frequency")
        ParameterValidator.assert_type_and_value(ignore_genes, bool, "EditDistanceEncoder", "ignore_genes")

        # todo infer executable path from somewhere (installed)
        matchairr_path = Path(matchairr_path)
        try:
            matchairr_result = subprocess.run([matchairr_path, "-h"], capture_output=True)
            assert matchairr_result.returncode == 0, "exit code was non-zero."
        except Exception as e:
            raise Exception(f"EditDistanceEncoder: failed to call MatchAIRR: {e}\n"
                            f"Please ensure MatchAIRR has been correctly installed and is available at {matchairr_path}.")

        return {
            "matchairr_path": matchairr_path,
            "distance_metric": distance_metric,
            "differences": differences,
            "indels": indels,
            "ignore_frequency": ignore_frequency,
            "ignore_genes": ignore_genes,
            "context": context,
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = EditDistanceEncoder._prepare_parameters(**params)
            return EditDistanceEncoder(**prepared_params)
        else:
            raise ValueError("EditDistanceEncoder is not defined for dataset types which are not RepertoireDataset.")


    def build_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:

        lbl = params.label_config.get_labels_by_name()
        tmp_labels = dataset.get_metadata(lbl, return_df=True)

        return tmp_labels.to_dict("list")


    def encode(self, dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        train_repertoire_ids = EncoderHelper.prepare_training_ids(dataset, params)
        labels = self.build_labels(dataset, params) if params.encode_labels else None

        distance_matrix = self.build_distance_matrix(dataset, params, train_repertoire_ids)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix, labels=labels, example_ids=distance_matrix.index.values,
                                                   encoding=EditDistanceEncoder.__name__)
        return encoded_dataset

    def build_distance_matrix(self, dataset: RepertoireDataset, params: EncoderParams, train_repertoire_ids: list):
        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]
        raw_distance_matrix, repertoire_sizes = self._run_matchairr(current_dataset, params)

        distance_matrix = self.apply_distance_fn(raw_distance_matrix, repertoire_sizes)

        repertoire_ids = dataset.get_repertoire_ids()

        distance_matrix = distance_matrix.loc[repertoire_ids, train_repertoire_ids]

        return distance_matrix

    def apply_distance_fn(self, raw_distance_matrix, repertoire_sizes):
        distance_matrix = pd.DataFrame().reindex_like(raw_distance_matrix)

        for rowIndex, row in distance_matrix.iterrows():
            for columnIndex, value in row.items():
                distance_matrix.loc[rowIndex, columnIndex] = self.jaccard_dist(repertoire_sizes[rowIndex],
                                                                                repertoire_sizes[columnIndex],
                                                                                raw_distance_matrix.loc[rowIndex, columnIndex])
        return distance_matrix

    def jaccard_dist(self, rep_1_size, rep_2_size, intersect):
        return 1 - intersect / (rep_1_size + rep_2_size - intersect)

    def _run_matchairr(self, dataset: RepertoireDataset, params: EncoderParams):
        repertoire_sizes = {}


        testfile = params.result_path / "test.tsv"

        with NamedTemporaryFile(mode='w') as tmp:
            for repertoire in dataset.get_data():
                repertoire_contents = repertoire.get_attributes([EnvironmentSettings.get_sequence_type().value, "counts", "v_genes", "j_genes"])
                repertoire_contents = pd.DataFrame({**repertoire_contents, "identifier": repertoire.identifier})

                # todo deal with v/j or counts missing if not specified
                na_rows = sum(repertoire_contents.isnull().any(axis=1))
                repertoire_contents.dropna(inplace=True)
                if na_rows > 0:
                    warnings.warn(f"EditDistanceEncoder: removed {na_rows} entries from repertoire {repertoire.identifier} due to missing values.")

                repertoire_sizes[repertoire.identifier] = sum(repertoire_contents["counts"].astype(int))

                print(f"result written to {testfile}")
                repertoire_contents.to_csv(testfile, mode='a', header=False, index=False, sep="\t")
                repertoire_contents.to_csv(tmp.name, mode='a', header=False, index=False, sep="\t")
            args = self._get_cmd_args(tmp.name, params.pool_size)
            matchairr_result = subprocess.run(args, capture_output=True, text=True)

        print("****stdout")
        print(matchairr_result.stdout)
        print("****stderr")
        print(matchairr_result.stderr)

        if matchairr_result.stdout == "":
            raise RuntimeError(f"EditDistanceEncoder: failed to calculate the distance matrix with MatchAIRR, the following error occurred:\n\n{matchairr_result.stderr}")

        raw_distance_matrix = pd.read_csv(StringIO(matchairr_result.stdout), sep="\t", index_col=0)

        print("raw dist matrix")
        print(raw_distance_matrix)

        return raw_distance_matrix, repertoire_sizes

    def _get_cmd_args(self, filename, number_of_processes):
        indels_args = ["-i"] if self.indels else []
        frequency_args = ["-f"] if self.ignore_frequency else []
        ignore_genes = ["-g"] if self.ignore_genes else []

        number_of_processes = 1 if number_of_processes < 1 else number_of_processes

        return [str(self.matchairr_path), "-m", "-d", str(self.differences), "-t", str(number_of_processes)] + \
               indels_args + frequency_args + ignore_genes + [filename, filename]

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        # todo deal with this??
        # encoder.comparison = UtilIO.import_comparison_data(encoder_file.parent)
        return encoder

    @staticmethod
    def get_documentation():
        doc = str(EditDistanceEncoder.__doc__)

        valid_values = [metric.name for metric in DistanceMetricType]
        valid_values = str(valid_values)[1:-1].replace("'", "`")
        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Names of different distance metric types are allowed values in the specification.": f"Valid values are: {valid_values}.",
            "Valid values include any repertoire attribute (sequence, amino acid sequence, V gene etc).":
                f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
