import copy
import subprocess
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List

import math
import numpy as np
import pandas as pd

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.abundance_encoding.AbundanceEncoderHelper import AbundanceEncoderHelper
from immuneML.encodings.abundance_encoding.CompAIRRBatchIterator import CompAIRRBatchIterator
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class CompAIRRSequenceAbundanceEncoder(DatasetEncoder):
    """
    This encoder works similarly to the :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`,
    but internally uses `CompAIRR <https://github.com/uio-bmi/compairr/>`_ to accelerate core computations.

    This encoder represents the repertoires as vectors where:

    - the first element corresponds to the number of label-associated clonotypes
    - the second element is the total number of unique clonotypes

    To determine what clonotypes (amino acid sequences with or without matching V/J genes) are label-associated, Fisher's exact test (one-sided)
    is used.

    The encoder also writes out files containing the contingency table used for fisher's exact test,
    the resulting p-values, and the significantly abundant sequences
    (use :py:obj:`~immuneML.reports.encoding_reports.RelevantSequenceExporter.RelevantSequenceExporter` to export these sequences in AIRR format).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

    Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
    in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
    See :ref:`Reproduction of the CMV status predictions study` for an example using :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder`.

    **Specification arguments:**

    - p_value_threshold (float): The p value threshold to be used by the statistical test.

    - compairr_path (Path): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR
      has been installed such that it can be called directly on the command line with the command 'compairr',
      or that it is located at /usr/local/bin/compairr.

    - ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains
      have to match. If True, gene information is ignored. By default, ignore_genes is False.

    - sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
      This does not affect the results of the encoding, but may affect the speed and memory usage. The default value is 1.000.000

    - threads (int): The number of threads to use for parallelization. This does not affect the results of the encoding, only the speed.
      The default number of threads is 8.

    - keep_temporary_files (bool): whether to keep temporary files, including CompAIRR input, output and log files, and the sequence
      presence matrix. This may take a lot of storage space if the input dataset is large. By default, temporary files are not kept.


    **YAML specification:**

    .. code-block:: yaml

        definitions:
            encodings:
                my_sa_encoding:
                    CompAIRRSequenceAbundance:
                        compairr_path: optional/path/to/compairr
                        p_value_threshold: 0.05
                        ignore_genes: False
                        threads: 8

    """

    RELEVANT_SEQUENCE_ABUNDANCE = "relevant_sequence_abundance"
    TOTAL_SEQUENCE_ABUNDANCE = "total_sequence_abundance"
    OUTPUT_FILENAME = "compairr_out.tsv"
    LOG_FILENAME = "compairr_log.txt"

    def __init__(self, p_value_threshold: float, compairr_path: str, sequence_batch_size: int, ignore_genes: bool, keep_temporary_files: bool,
                 threads: int, name: str = None):
        super().__init__(name=name)
        self.p_value_threshold = p_value_threshold
        self.sequence_batch_size = sequence_batch_size
        self.keep_temporary_files = keep_temporary_files

        self.repertoires_filepath = None
        self.sequences_filepaths = None
        self.relevant_indices_path = None
        self.relevant_sequence_path = None
        self.contingency_table_path = None
        self.p_values_path = None
        self.context = None
        self.compairr_sequence_presence = None

        self.compairr_params = CompAIRRParams(compairr_path=Path(compairr_path),
                                              keep_compairr_input=True,
                                              differences=0,
                                              indels=False,
                                              ignore_counts=True,
                                              ignore_genes=ignore_genes,
                                              threads=threads,
                                              output_filename=None,
                                              log_filename=None,
                                              output_pairs=False,
                                              pairs_filename=None)

    @staticmethod
    def _prepare_parameters(p_value_threshold: float, compairr_path: str, sequence_batch_size: int, ignore_genes: bool, keep_temporary_files: bool,
                            threads: int,
                            name: str = None):
        ParameterValidator.assert_type_and_value(p_value_threshold, float, "CompAIRRSequenceAbundanceEncoder", "p_value_threshold", min_inclusive=0,
                                                 max_inclusive=1)
        ParameterValidator.assert_type_and_value(sequence_batch_size, int, "CompAIRRSequenceAbundanceEncoder", "sequence_batch_size", min_inclusive=1)
        ParameterValidator.assert_type_and_value(ignore_genes, bool, "CompAIRRSequenceAbundanceEncoder", "ignore_genes")
        ParameterValidator.assert_type_and_value(keep_temporary_files, bool, "CompAIRRSequenceAbundanceEncoder", "keep_temporary_files")
        ParameterValidator.assert_type_and_value(threads, int, "CompAIRRSequenceAbundanceEncoder", "threads", min_inclusive=1)

        compairr_path = CompAIRRHelper.determine_compairr_path(compairr_path)

        return {
            "p_value_threshold": p_value_threshold,
            "compairr_path": compairr_path,
            "sequence_batch_size": sequence_batch_size,
            "ignore_genes": ignore_genes,
            "keep_temporary_files": keep_temporary_files,
            "threads": threads,
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "CompAIRRSequenceAbundanceEncoder: this encoding only works on repertoire datasets."
        prepared_params = CompAIRRSequenceAbundanceEncoder._prepare_parameters(**params)
        return CompAIRRSequenceAbundanceEncoder(**prepared_params)

    def encode(self, dataset, params: EncoderParams):
        EncoderHelper.check_positive_class_labels(params.label_config, CompAIRRSequenceAbundanceEncoder.__name__)
        self.compairr_params.is_cdr3 = dataset.repertoires[0].get_region_type() == RegionType.IMGT_CDR3

        self.compairr_sequence_presence = self._prepare_sequence_presence_data(dataset, params)

        return self._encode_data(dataset, params)

    def _prepare_sequence_presence_data(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)

        full_sequence_set = self._get_full_sequence_set(full_dataset)
        compairr_sequence_presence = self._get_sequence_presence(full_dataset, full_sequence_set, params)

        return compairr_sequence_presence

    def _get_full_sequence_set(self, full_dataset):
        full_sequence_set = CacheHandler.memo_by_params(self._build_dataset_params(full_dataset),
                                                        lambda: self.get_sequence_set(full_dataset))

        return full_sequence_set

    def get_sequence_set(self, repertoire_dataset):
        attributes = self.get_relevant_sequence_attributes()
        sequence_set = set()

        for repertoire in repertoire_dataset.get_data():
            sequence_set.update(self.get_sequence_set_for_repertoire(repertoire, attributes))

        return np.array(list(sequence_set))

    def get_sequence_set_for_repertoire(self, repertoire, sequence_attributes):
        return set(zip(*[value for value in repertoire.get_attributes(sequence_attributes, True).values() if value is not None]))

    def _get_sequence_presence(self, full_dataset, full_sequence_set, params):
        compairr_sequence_presence = CacheHandler.memo_by_params(
            self._build_sequence_presence_params(full_dataset, self.compairr_params),
            lambda: self._compute_sequence_presence_with_compairr(full_dataset, full_sequence_set, params))

        return compairr_sequence_presence

    def _build_sequence_presence_params(self, dataset, compairr_params):
        return (("dataset_identifier", dataset.identifier),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids())),
                ("ignore_genes", compairr_params.ignore_genes),
                ("indels", compairr_params.indels),
                ("differences", compairr_params.differences),
                ("ignore_counts", compairr_params.ignore_counts))

    def _build_dataset_params(self, dataset):
        return (("dataset_identifier", dataset.identifier),
                ("repertoire_ids", tuple(dataset.get_repertoire_ids())),
                "sequence_attributes", tuple(self.get_relevant_sequence_attributes()))

    def _compute_sequence_presence_with_compairr(self, dataset, full_sequence_set, params):
        self._prepare_compairr_input_files(dataset, full_sequence_set, params.result_path)

        arguments = [(sequences_filepath, params.result_path) for sequences_filepath in self.sequences_filepaths]

        with Pool(params.pool_size) as pool:
            paths = pool.starmap(self._new_run_compairr_on_batch, arguments)

        if not self.keep_temporary_files:
            self._remove_temporary_files(params.result_path)

        return CompAIRRBatchIterator(paths, self.sequence_batch_size)

    def _prepare_compairr_input_files(self, dataset, full_sequence_set, result_path):
        PathBuilder.build(result_path)

        if self.repertoires_filepath is None:
            self.repertoires_filepath = result_path / "compairr_repertoires.tsv"

        CompAIRRHelper.write_repertoire_file(dataset, self.repertoires_filepath, self.compairr_params)

        if self.sequences_filepaths is None:
            self.sequences_filepaths = []

            n_sequence_files = math.ceil(len(full_sequence_set) / self.sequence_batch_size)
            subset_start_index = 0
            subset_end_index = min(self.sequence_batch_size, len(full_sequence_set))

            for file_index in range(n_sequence_files):
                filename = result_path / f"compairr_sequences_batch{file_index}.tsv"
                self.sequences_filepaths.append(filename)
                sequence_subset = full_sequence_set[subset_start_index:subset_end_index]

                self.write_sequence_set_file(sequence_subset, filename, offset=subset_start_index, region_type=dataset.repertoires[0].get_region_type())

                subset_start_index += self.sequence_batch_size
                subset_end_index = min(subset_end_index + self.sequence_batch_size, len(full_sequence_set))

    def _remove_temporary_files(self, result_path):
        self.repertoires_filepath.unlink()

        for file in self.sequences_filepaths:
            file.unlink()

    def write_sequence_set_file(self, sequence_set, filename, offset=0, region_type=RegionType.IMGT_JUNCTION):
        sequence_col = 'junction' if region_type == RegionType.IMGT_JUNCTION else 'cdr3'
        if EnvironmentSettings.get_sequence_type() == SequenceType.AMINO_ACID:
            sequence_col = f"{sequence_col}_aa"
        vj_header = "" if self.compairr_params.ignore_genes else "\tv_call\tj_call"

        with open(filename, "w") as file:
            file.write(f"{sequence_col}{vj_header}\tduplicate_count\trepertoire_id\n")

            for id, sequence_info in enumerate(sequence_set, offset):
                file.write("\t".join(sequence_info) + f"\t1\t{id}\n")

    def _run_compairr_on_batch(self, sequences_filepath, result_path):
        batch = sequences_filepath.stem.split("_")[-1]
        compairr_params = copy.copy(self.compairr_params)
        compairr_params.output_filename = f"compairr_out_{batch}.txt"
        compairr_params.log_filename = f"compairr_log_{batch}.txt"

        args = CompAIRRHelper.get_cmd_args(compairr_params, [sequences_filepath, self.repertoires_filepath], result_path)
        compairr_result = subprocess.run(args, capture_output=True, text=True)
        return CompAIRRHelper.process_compairr_output_file(compairr_result, compairr_params, result_path)

    def _new_run_compairr_on_batch(self, sequences_filepath, result_path):
        batch = sequences_filepath.stem.split("_")[-1]
        compairr_params = copy.copy(self.compairr_params)
        compairr_params.output_filename = f"compairr_out_{batch}.txt"
        compairr_params.log_filename = f"compairr_log_{batch}.txt"

        args = CompAIRRHelper.get_cmd_args(compairr_params, [sequences_filepath, self.repertoires_filepath], result_path)
        compairr_result = subprocess.run(args, capture_output=True, text=True)
        return CompAIRRHelper.verify_compairr_output_path(compairr_result, compairr_params, result_path)

    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams):
        label = params.label_config.get_label_objects()[0]

        examples = self._calculate_abundance_matrix(dataset, self.compairr_sequence_presence, params)

        encoded_data = EncodedData(examples, dataset.get_metadata([label.name]) if params.encode_labels else None, dataset.get_repertoire_ids(),
                                   [CompAIRRSequenceAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE,
                                    CompAIRRSequenceAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   example_weights=dataset.get_example_weights(),
                                   encoding=CompAIRRSequenceAbundanceEncoder.__name__,
                                   info={"relevant_sequence_path": self.relevant_sequence_path,
                                         "contingency_table_path": self.contingency_table_path,
                                         "p_values_path": self.p_values_path})

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    def _calculate_abundance_matrix(self, dataset: RepertoireDataset, compairr_sequence_presence, params: EncoderParams):
        repertoire_ids = dataset.get_repertoire_ids()
        compairr_sequence_presence.set_repertoire_ids(repertoire_ids)

        is_positive_class = AbundanceEncoderHelper.check_is_positive_class(dataset, repertoire_ids, params.label_config)

        relevant_sequence_indices, file_paths = AbundanceEncoderHelper\
            .get_relevant_sequence_indices(compairr_sequence_presence, is_positive_class, self.p_value_threshold, self.relevant_indices_path, params,
                                           cache_params=self._build_sequence_presence_params(dataset, self.compairr_params))
        self._write_relevant_sequences_csv(dataset, relevant_sequence_indices, params.result_path)
        self._set_file_paths(file_paths)

        abundance_matrix = self._build_abundance_matrix(compairr_sequence_presence, repertoire_ids, relevant_sequence_indices)

        return abundance_matrix

    def _build_abundance_matrix(self, compairr_sequence_presence, repertoire_ids, sequence_p_values_indices):
        abundance_matrix = np.zeros((len(repertoire_ids), 2))

        batch_start = 0

        for batch in compairr_sequence_presence.get_batches(repertoire_ids):
            batch_end = min(batch_start + self.sequence_batch_size, len(sequence_p_values_indices))

            partial_sequence_p_values_indices = sequence_p_values_indices[batch_start:batch_end]

            for idx, repertoire_id in enumerate(repertoire_ids):
                partial_repertoire_vector = batch[repertoire_id].to_numpy()

                relevant_sequence_abundance = np.sum(
                    partial_repertoire_vector[np.logical_and(partial_sequence_p_values_indices, partial_repertoire_vector)])
                total_sequence_abundance = np.sum(partial_repertoire_vector)
                abundance_matrix[idx] += [relevant_sequence_abundance, total_sequence_abundance]

            batch_start = batch_start + self.sequence_batch_size

        return abundance_matrix

    def _set_file_paths(self, file_paths):
        self.relevant_indices_path = file_paths["relevant_indices_path"]
        self.contingency_table_path = file_paths["contingency_table_path"] if "contingency_table_path" in file_paths else None
        self.p_values_path = file_paths["p_values_path"] if "p_values_path" in file_paths else None

    def _write_relevant_sequences_csv(self, dataset, relevant_sequence_indices, result_path):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        full_sequence_set = self._get_full_sequence_set(full_dataset)
        relevant_sequences = full_sequence_set[relevant_sequence_indices]

        if self.relevant_sequence_path is None:
            self.relevant_sequence_path = result_path / 'relevant_sequences.csv'

        df = pd.DataFrame(relevant_sequences, columns=self.get_relevant_sequence_attributes())
        df.to_csv(self.relevant_sequence_path, sep=",", index=False)

    def get_relevant_sequence_attributes(self):
        attributes = [EnvironmentSettings.get_sequence_type().value]

        if not self.compairr_params.ignore_genes:
            attributes += ["v_call", "j_call"]

        return attributes

    def set_context(self, context: dict):
        self.context = context
        return self

    def get_additional_files(self) -> List[Path]:
        return [file for file in [self.relevant_indices_path, self.relevant_sequence_path, self.contingency_table_path, self.p_values_path] if file]

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.relevant_indices_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_indices_path")

        return encoder
