import subprocess
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


class CompAIRRDistanceEncoder(DatasetEncoder):
    """
    Encodes a given RepertoireDataset as a distance matrix, using the Morisita-Horn distance metric.
    Internally, `CompAIRR <https://github.com/uio-bmi/compairr/>`_ is used for fast calculation of overlap between repertoires.
    This creates a pairwise distance matrix between each of the repertoires.
    The distance is calculated based on the number of matching receptor chain sequences between the repertoires. This matching may be
    defined to permit 1 or 2 mismatching amino acid/nucleotide positions and 1 indel in the sequence. Furthermore,
    matching may or may not include V and J gene information, and sequence frequencies may be included or ignored.

    When mismatches (differences and indels) are allowed, the Morisita-Horn similarity may exceed 1. In this case, the
    Morisita-Horn distance (= similarity - 1) is set to 0 to avoid negative distance scores.


    **Specification arguments:**

    - compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR has been
      installed such that it can be called directly on the command line with the command 'compairr', or that it is
      located at /usr/local/bin/compairr.

    - keep_compairr_input (bool): whether to keep the input file that was passed to CompAIRR. This may take a lot of
      storage space if the input dataset is large. By default, the input file is not kept.

    - differences (int): Number of differences allowed between the sequences of two immune receptor chains, this may be
      between 0 and 2. By default, differences is 0.

    - indels (bool): Whether to allow an indel. This is only possible if differences is 1. By default, indels is False.

    - ignore_counts (bool): Whether to ignore the frequencies of the immune receptor chains. If False, frequencies will
      be included, meaning the 'counts' values for the receptors available in two repertoires are multiplied. If False,
      only the number of unique overlapping immune receptors ('clones') are considered. By default, ignore_counts is False.

    - ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor
      chains have to match. If True, gene information is ignored. By default, ignore_genes is False.

    - threads (int): The number of threads to use for parallelization. Default is 8.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_distance_encoder:
                    CompAIRRDistance:
                        compairr_path: optional/path/to/compairr
                        differences: 0
                        indels: False
                        ignore_counts: False
                        ignore_genes: False

    """

    OUTPUT_FILENAME = "compairr_results.txt"
    INPUT_FILENAME = "compairr_input.tsv"
    LOG_FILENAME = "compairr_log.txt"

    def __init__(self, compairr_path: Path, keep_compairr_input: bool, differences: int, indels: bool, ignore_counts: bool, ignore_genes: bool,
                 threads: int, context: dict = None, name: str = None):
        self.compairr_params = CompAIRRParams(compairr_path=Path(compairr_path),
                                              keep_compairr_input=keep_compairr_input,
                                              differences=differences,
                                              indels=indels,
                                              ignore_counts=ignore_counts,
                                              ignore_genes=ignore_genes,
                                              threads=threads,
                                              output_filename=CompAIRRDistanceEncoder.OUTPUT_FILENAME,
                                              log_filename=CompAIRRDistanceEncoder.LOG_FILENAME, output_pairs=False, pairs_filename=None)

        self.context = context
        self.name = name

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def _prepare_parameters(compairr_path: str, keep_compairr_input: bool, differences: int, indels: bool, ignore_counts: bool, ignore_genes: bool,
                            threads: int, context: dict = None, name: str = None):
        ParameterValidator.assert_type_and_value(differences, int, "CompAIRRDistanceEncoder", "differences", min_inclusive=0, max_inclusive=2)
        ParameterValidator.assert_type_and_value(indels, bool, "CompAIRRDistanceEncoder", "indels")
        if indels:
            assert differences == 1, f"CompAIRRDistanceEncoder: If indels is True, differences is only allowed to be 1, found {differences}"

        ParameterValidator.assert_type_and_value(ignore_counts, bool, "CompAIRRDistanceEncoder", "ignore_counts")
        ParameterValidator.assert_type_and_value(ignore_genes, bool, "CompAIRRDistanceEncoder", "ignore_genes")
        ParameterValidator.assert_type_and_value(threads, int, "CompAIRRDistanceEncoder", "threads", min_inclusive=1)
        ParameterValidator.assert_type_and_value(keep_compairr_input, bool, "CompAIRRDistanceEncoder", "keep_compairr_input")

        compairr_path = CompAIRRHelper.determine_compairr_path(compairr_path)

        return {
            "compairr_path": compairr_path,
            "keep_compairr_input": keep_compairr_input,
            "differences": differences,
            "indels": indels,
            "ignore_counts": ignore_counts,
            "ignore_genes": ignore_genes,
            "threads": threads,
            "context": context,
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = CompAIRRDistanceEncoder._prepare_parameters(**params)
            return CompAIRRDistanceEncoder(**prepared_params)
        else:
            raise ValueError("CompAIRRDistanceEncoder is not defined for dataset types which are not RepertoireDataset.")

    def build_labels(self, dataset: RepertoireDataset, params: EncoderParams) -> dict:
        lbl = params.label_config.get_labels_by_name()
        return dataset.get_metadata(lbl, return_df=False)

    def encode(self, dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        train_repertoire_ids = EncoderHelper.prepare_training_ids(dataset, params)
        labels = self.build_labels(dataset, params) if params.encode_labels else None

        distance_matrix = self.build_distance_matrix(dataset, params, train_repertoire_ids)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=distance_matrix.to_numpy(),
                                                   labels=labels,
                                                   feature_names=distance_matrix.columns.values,
                                                   example_ids=distance_matrix.index.values,
                                                   example_weights=EncoderHelper.get_example_weights_by_identifiers(dataset, distance_matrix.index.values),
                                                   encoding=CompAIRRDistanceEncoder.__name__)
        return encoded_dataset

    def build_distance_matrix(self, dataset: RepertoireDataset, params: EncoderParams, train_repertoire_ids: list):
        current_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]
        raw_distance_matrix, repertoire_sizes, repertoire_indices = self._compute_overlap_with_compairr(current_dataset, params)

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

        if mh_distance < -0.3 and self.compairr_params.differences == 0:
            warnings.warn(
                f"CompAIRRDistanceEncoder: Morisita-Horn similarity can only be in the range [0, 1], found {mh_similarity} "
                f"when comparing repertoires {rowIndex} and {columnIndex}.")

        if mh_distance < 0:
            warnings.warn(
                f"CompAIRRDistanceEncoder: found negative distance {mh_distance} when comparing repertoires {rowIndex} and {columnIndex}, "
                f"distance will be set to 0.")
            mh_distance = 0

        return mh_distance

    def _compute_overlap_with_compairr(self, dataset: RepertoireDataset, params: EncoderParams):

        if self.compairr_params.keep_compairr_input:
            raw_distance_matrix, repertoire_sizes, repertoire_indices = self._run_compairr(dataset, params,
                                                                                           params.result_path / CompAIRRDistanceEncoder.INPUT_FILENAME)
        else:
            with NamedTemporaryFile(mode='w') as tmp:
                raw_distance_matrix, repertoire_sizes, repertoire_indices = self._run_compairr(dataset, params, tmp.name)

        return raw_distance_matrix, repertoire_sizes, repertoire_indices

    def _run_compairr(self, dataset, params, filename):
        repertoire_sizes, repertoire_indices = self._prepare_repertoire_file(dataset, filename)

        self.compairr_params.is_cdr3 = dataset.repertoires[0].get_region_type() == RegionType.IMGT_CDR3
        args = CompAIRRHelper.get_cmd_args(self.compairr_params, [filename], params.result_path)
        compairr_result = subprocess.run(args, capture_output=True, text=True)

        raw_distance_matrix = CompAIRRHelper.process_compairr_output_file(compairr_result, self.compairr_params, params.result_path)

        return raw_distance_matrix, repertoire_sizes, repertoire_indices

    def _prepare_repertoire_file(self, dataset, filename):
        repertoire_sizes = {}
        repertoire_indices = {}

        mode = "w"
        header = True

        for repertoire in dataset.get_data():
            repertoire_contents = CompAIRRHelper.get_repertoire_contents(repertoire, self.compairr_params)

            repertoire_counts = repertoire_contents["duplicate_count"].astype(int)

            repertoire_sizes[repertoire.identifier] = sum(repertoire_counts)
            repertoire_indices[repertoire.identifier] = sum(np.square(repertoire_counts)) / np.square(sum(repertoire_counts))

            repertoire_contents.to_csv(filename, mode=mode, header=header, index=False, sep="\t")

            mode = "a"
            header = False

        return repertoire_sizes, repertoire_indices

