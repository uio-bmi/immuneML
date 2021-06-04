from pathlib import Path
import subprocess
from io import StringIO
import pandas as pd
import numpy as np
import warnings
from tempfile import NamedTemporaryFile


from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


class EditDistanceEncoder(DatasetEncoder):
    """
    Encodes a given RepertoireDataset as a distance matrix, using the Morisita-Horn distance metric.
    Internally, `CompAIRR <https://github.com/uio-bmi/vdjsearch/>`_ is used for fast calculation of overlap between repertoires.
    This creates a pairwise distance matrix between each of the repertoires.
    The distance is calculated based on the number of matching receptor chain sequences between the repertoires. This matching may be
    defined to permit 1 or 2 mismatching amino acid/nucleotide positions and 1 indel in the sequence. Furthermore,
    matching may or may not include V and J gene information, and sequence frequencies may be included or ignored.

    When mismatches (differences and indels) are allowed, the Morisita-Horn similarity may exceed 1. In this case, the
    Morisita-Horn distance (= similarity - 1) is set to 0 to avoid negative distance scores.


    Arguments:

        compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR
        has been installed such that it can be called directly on the command line, or that it is located at /usr/local/bin/compairr.

        keep_compairr_input (bool): whether to keep the input file that was passed to CompAIRR. This may take a lot of
        storage space if the input dataset is large.

        differences (int): Number of differences allowed between the sequences of two immune receptor chains, this
        may be between 0 and 2. By default, differences is 0.

        indels (bool): Whether to allow an indel. This is only possible if differences is 1. By default, indels is False.

        ignore_counts (bool): Whether to ignore the frequencies of the immune receptor chains. If False, frequencies
        will be included, meaning the 'counts' values for the receptors available in two repertoires are multiplied.
        If False, only the number of unique overlapping immune receptors ('clones') are considered.
        By default, ignore_counts is False.

        ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains
        have to match. If True, gene information is ignored. By default, ignore_genes is False.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_distance_encoder:
            Distance:
                compairr_path: optional/path/to/compairr
                differences: 0
                indels: False
                ignore_counts: False
                ignore_genes: False

    """

    OUTPUT_FILENAME = "compairr_results.txt"
    INPUT_FILENAME = "compairr_input.tsv"
    LOG_FILENAME = "compairr_log.txt"

    def __init__(self, compairr_path: Path, keep_compairr_input: bool, differences: int, indels: bool, ignore_counts: bool, ignore_genes: bool, context: dict = None, name: str = None):
        self.compairr_path = Path(compairr_path)
        self.keep_compairr_input = keep_compairr_input
        self.differences = differences
        self.indels = indels
        self.ignore_counts = ignore_counts
        self.ignore_genes = ignore_genes
        self.context = context
        self.name = name
        self.comparison = None

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def _prepare_parameters(compairr_path: str, keep_compairr_input: bool, differences: int, indels: bool, ignore_counts: bool, ignore_genes: bool, context: dict = None, name: str = None):
        ParameterValidator.assert_type_and_value(differences, int, "EditDistanceEncoder", "differences", min_inclusive=0, max_inclusive=2)
        ParameterValidator.assert_type_and_value(indels, bool, "EditDistanceEncoder", "indels")
        if indels:
            assert differences == 1, f"EditDistanceEncoder: If indels is True, differences is only allowed to be 1, found {differences}"

        ParameterValidator.assert_type_and_value(ignore_counts, bool, "EditDistanceEncoder", "ignore_counts")
        ParameterValidator.assert_type_and_value(ignore_genes, bool, "EditDistanceEncoder", "ignore_genes")
        ParameterValidator.assert_type_and_value(keep_compairr_input, bool, "EditDistanceEncoder", "keep_compairr_input")

        compairr_path = EditDistanceEncoder._determine_compairr_path(compairr_path)

        return {
            "compairr_path": compairr_path,
            "keep_compairr_input": keep_compairr_input,
            "differences": differences,
            "indels": indels,
            "ignore_counts": ignore_counts,
            "ignore_genes": ignore_genes,
            "context": context,
            "name": name
        }

    @staticmethod
    def _determine_compairr_path(compairr_path):
        if compairr_path is None:
            try:
                compairr_path = EditDistanceEncoder._check_compairr_path("compairr")
            except Exception as e:
                compairr_path = EditDistanceEncoder._check_compairr_path("/usr/local/bin/compairr")
        else:
            compairr_path = EditDistanceEncoder._check_compairr_path(compairr_path)

        return compairr_path


    @staticmethod
    def _check_compairr_path(compairr_path):
        try:
            compairr_result = subprocess.run([str(Path(compairr_path)), "--version"], capture_output=True)
            assert compairr_result.returncode == 0, "exit code was non-zero."
            output = str(compairr_result.stderr).split()
            major, minor, patch = output[1].split(".")
            assert int(major) >= 1, f"CompAIRR version 1 or higher is required, found version {output[1]}"
        except Exception as e:
            raise Exception(f"EditDistanceEncoder: failed to call CompAIRR: {e}\n"
                            f"Please ensure the correct version of CompAIRR has been installed, or provide the path to the CompAIRR executable.")

        return compairr_path


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
        raw_distance_matrix, repertoire_sizes, repertoire_indices = self._get_raw_overlap(current_dataset, params)

        distance_matrix = self._morisita_horn(raw_distance_matrix, repertoire_sizes, repertoire_indices)

        repertoire_ids = dataset.get_repertoire_ids()

        distance_matrix = distance_matrix.loc[repertoire_ids, train_repertoire_ids]

        return distance_matrix

    def _morisita_horn(self, raw_distance_matrix, repertoire_sizes, repertoire_indices):
        distance_matrix = pd.DataFrame().reindex_like(raw_distance_matrix)

        for rowIndex, row in distance_matrix.iterrows():
            for columnIndex, value in row.items():
                mh_similarity = (2 * raw_distance_matrix.loc[rowIndex, columnIndex]) / \
                                ((repertoire_indices[rowIndex] + repertoire_indices[columnIndex]) *
                                 (repertoire_sizes[rowIndex] * repertoire_sizes[columnIndex]))

                distance_matrix.loc[rowIndex, columnIndex] = self._check_distance(mh_similarity, rowIndex, columnIndex)

        return distance_matrix

    def _check_distance(self, mh_similarity, rowIndex, columnIndex):
        mh_distance = 1 - mh_similarity

        if mh_distance < -0.3 and self.differences == 0:
            warnings.warn(
                f"EditDistanceEncoder: Morisita-Horn similarity can only be in the range [0, 1], found {mh_similarity} "
                f"when comparing repertoires {rowIndex} and {columnIndex}.")

        if mh_distance < 0:
            warnings.warn(
                f"EditDistanceEncoder: found negative distance {mh_distance} when comparing repertoires {rowIndex} and {columnIndex}, "
                f"distance will be set to 0.")
            mh_distance = 0

        return mh_distance

    def _get_raw_overlap(self, dataset: RepertoireDataset, params: EncoderParams):

        if self.keep_compairr_input:
            compairr_result, repertoire_sizes, repertoire_indices = self._run_compairr(dataset, params, params.result_path / EditDistanceEncoder.INPUT_FILENAME)
        else:
            with NamedTemporaryFile(mode='w') as tmp:
                compairr_result, repertoire_sizes, repertoire_indices = self._run_compairr(dataset, params, tmp.name)

        output_file = params.result_path / EditDistanceEncoder.OUTPUT_FILENAME

        if not output_file.is_file():
            raise RuntimeError(f"EditDistanceEncoder: failed to calculate the distance matrix with CompAIRR. "
                               f"The following error occurred: {compairr_result.stderr}")

        raw_distance_matrix = pd.read_csv(output_file, sep="\t", index_col=0)

        return raw_distance_matrix, repertoire_sizes, repertoire_indices

    def _run_compairr(self, dataset, params, filename):
        repertoire_sizes = {}
        repertoire_indices = {}

        with open(filename, "w") as file:
            file.write("junction_aa\tduplicate_count\tv_call\tj_call\trepertoire_id\n")

        for repertoire in dataset.get_data():
            repertoire_contents = self._get_repertoire_contents(repertoire)

            repertoire_counts = repertoire_contents["counts"].astype(int)

            repertoire_sizes[repertoire.identifier] = sum(repertoire_counts)
            repertoire_indices[repertoire.identifier] = sum(np.square(repertoire_counts)) / np.square(
                sum(repertoire_counts))

            repertoire_contents.to_csv(filename, mode='a', header=False, index=False, sep="\t")

        args = self._get_cmd_args(filename, params.result_path, params.pool_size)
        compairr_result = subprocess.run(args, capture_output=True, text=True)

        return compairr_result, repertoire_sizes, repertoire_indices



    def _get_repertoire_contents(self, repertoire):
        repertoire_contents = repertoire.get_attributes([EnvironmentSettings.get_sequence_type().value, "counts", "v_genes", "j_genes"])
        repertoire_contents = pd.DataFrame({**repertoire_contents, "identifier": repertoire.identifier})

        check_na_rows = [EnvironmentSettings.get_sequence_type().value]
        check_na_rows += [] if self.ignore_counts else ["counts"]
        check_na_rows += [] if self.ignore_genes else ["v_genes", "j_genes"]

        n_rows_before = len(repertoire_contents)

        repertoire_contents.dropna(inplace=True, subset=check_na_rows)

        if n_rows_before > len(repertoire_contents):
            warnings.warn(
                f"EditDistanceEncoder: removed {n_rows_before - len(repertoire_contents)} entries from repertoire {repertoire.identifier} due to missing values.")

        if self.ignore_counts:
            repertoire_contents["counts"] = 1

        return repertoire_contents

    def _get_cmd_args(self, input_file, result_path, number_of_processes=1):
        indels_args = ["-i"] if self.indels else []
        frequency_args = ["-f"] if self.ignore_counts else []
        ignore_genes = ["-g"] if self.ignore_genes else []
        output_args = ["-o", str(result_path / EditDistanceEncoder.OUTPUT_FILENAME), "-l", str(result_path / EditDistanceEncoder.LOG_FILENAME)]

        number_of_processes = 1 if number_of_processes < 1 else number_of_processes

        return [str(self.compairr_path), "-m", "-d", str(self.differences), "-t", str(number_of_processes)] + \
               indels_args + frequency_args + ignore_genes + output_args + [input_file]

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        return encoder