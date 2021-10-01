from pathlib import Path
from typing import List
import subprocess

import pandas as pd
import numpy as np
import fisher
import pickle

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.encodings.filtered_sequence_encoding.SequenceFilterHelper import SequenceFilterHelper
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from scripts.specification_util import update_docs_per_mapping
from immuneML.util.CompAIRRHelper import CompAIRRHelper

class CompAIRRSequenceAbundanceEncoder(DatasetEncoder):
    """
    This encoder represents the repertoires as vectors where:

    - the first element corresponds to the number of label-associated clonotypes
    - the second element is the total number of unique clonotypes

    To determine what clonotypes (with features defined by comparison_attributes) are label-associated
    based on a statistical test. The statistical test used is Fisher's exact test (one-sided).

    Reference: Emerson, Ryan O. et al.
    ‘Immunosequencing Identifies Signatures of Cytomegalovirus Exposure History and HLA-Mediated Effects on the T Cell Repertoire’.
    Nature Genetics 49, no. 5 (May 2017): 659–65. `doi.org/10.1038/ng.3822 <https://doi.org/10.1038/ng.3822>`_.

    Note: to use this encoder, it is necessary to explicitly define the positive class for the label when defining the label
    in the instruction. With positive class defined, it can then be determined which sequences are indicative of the positive class.
    For full example of using this encoder, see :ref:`Reproduction of the CMV status predictions study`.

    Arguments:

        <compairr settings>

        p_value_threshold (float): The p value threshold to be used by the statistical test.

        sequence_batch_size (int): The number of sequences in a batch when comparing sequences across repertoires, typically 100s of thousands.
        This does not affect the results of the encoding, only the speed.

        repertoire_batch_size (int): How many repertoires will be loaded at once. This does not affect the result of the encoding, only the speed.
        This value is a trade-off between the number of repertoires that can fit the RAM at the time and loading time from disk.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sa_encoding:
            SequenceAbundance:
                p_value_threshold: 0.05
                sequence_batch_size: 100000
                repertoire_batch_size: 32

    """

    RELEVANT_SEQUENCE_ABUNDANCE = "relevant_sequence_abundance"
    TOTAL_SEQUENCE_ABUNDANCE = "total_sequence_abundance"
    OUTPUT_FILENAME = "compairr_out.tsv"
    LOG_FILENAME = "compairr_log.txt"

    def __init__(self, p_value_threshold: float, compairr_path: str, keep_compairr_input: bool, ignore_genes: bool, threads: int, name: str = None):
        # todo deal with keepcompairrinput param, maybe remove?
        self.name = name
        self.p_value_threshold = p_value_threshold

        self.relevant_indices_path = None
        self.relevant_sequence_csv_path = None
        self.repertoires_filepath = None
        self.sequences_filepath = None
        self.context = None

        self.compairr_params = CompAIRRParams(compairr_path=Path(compairr_path),
                                              keep_compairr_input=keep_compairr_input,
                                              differences=0,
                                              indels=False,
                                              ignore_counts=True,
                                              ignore_genes=ignore_genes,
                                              threads=threads,
                                              output_filename=CompAIRRSequenceAbundanceEncoder.OUTPUT_FILENAME,
                                              log_filename=CompAIRRSequenceAbundanceEncoder.LOG_FILENAME)

        self.full_sequence_set = None
        self.raw_distance_matrix_np = None

    @staticmethod
    def _prepare_parameters(p_value_threshold: float, compairr_path: str, keep_compairr_input: bool, ignore_genes: bool, threads: int, name: str = None):
        ParameterValidator.assert_type_and_value(p_value_threshold, float, "CompAIRRSequenceAbundanceEncoder", "differences", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(ignore_genes, bool, "CompAIRRSequenceAbundanceEncoder", "ignore_genes")
        ParameterValidator.assert_type_and_value(threads, int, "CompAIRRSequenceAbundanceEncoder", "threads", min_inclusive=1)
        ParameterValidator.assert_type_and_value(keep_compairr_input, bool, "CompAIRRSequenceAbundanceEncoder", "keep_compairr_input")

        compairr_path = CompAIRRHelper.determine_compairr_path(compairr_path)

        return {
            "p_value_threshold": p_value_threshold,
            "compairr_path": compairr_path,
            "keep_compairr_input": keep_compairr_input,
            "ignore_genes": ignore_genes,
            "threads": threads,
            "name": name
        }

    @staticmethod
    def build_object(dataset, **params):
        assert isinstance(dataset, RepertoireDataset), "CompAIRRSequenceAbundanceEncoder: this encoding only works on repertoire datasets."
        prepared_params = CompAIRRSequenceAbundanceEncoder._prepare_parameters(**params)
        return CompAIRRSequenceAbundanceEncoder(**prepared_params)


    def encode(self, dataset, params: EncoderParams):
        self._check_label(params)
        self._prepare_sequence_presence_data(dataset, params)
        encoded_dataset = self._encode_data(dataset, params)

        return encoded_dataset

    def _check_label(self, params: EncoderParams):
        labels = params.label_config.get_label_objects()

        assert len(labels) == 1, \
            "CompAIRRSequenceAbundanceEncoder: this encoding works only for single label."

        assert isinstance(labels[0], Label) and labels[0].positive_class is not None and labels[0].positive_class != "", \
            f"{CompAIRRSequenceAbundanceEncoder.__name__}: to use this encoder, in the label definition in the specification of the instruction, define " \
            f"the positive class for the label. Now it is set to '{labels[0].positive_class}'. See documentation for this encoder for more details."

    def _prepare_sequence_presence_data(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)

        full_sequence_set = self._get_full_sequence_set(full_dataset)
        sequence_presence_matrix, matrix_repertoire_ids = self._get_sequence_presence(full_dataset, full_sequence_set, params)

        self.full_sequence_set = full_sequence_set
        self.sequence_presence_matrix = sequence_presence_matrix
        self.matrix_repertoire_ids = matrix_repertoire_ids

    def _get_full_sequence_set(self, full_dataset):
        full_sequence_set = CacheHandler.memo_by_params(self._build_dataset_params(full_dataset),
                                                        lambda: self.get_sequence_set(full_dataset))

        return full_sequence_set

    def _get_sequence_presence(self, full_dataset, full_sequence_set, params):
        sequence_presence_matrix, matrix_repertoire_ids = CacheHandler.memo_by_params(self._build_sequence_presence_params(full_dataset, self.compairr_params),
                                                                                      lambda: self._compute_sequence_presence_with_compairr(full_dataset, full_sequence_set, params))

        return sequence_presence_matrix, matrix_repertoire_ids


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

        args = CompAIRRHelper.get_cmd_args(self.compairr_params, [self.sequences_filepath, self.repertoires_filepath],
                                           params.result_path)
        compairr_result = subprocess.run(args, capture_output=True, text=True)
        sequence_presence_matrix = CompAIRRHelper.process_compairr_output_file(compairr_result, self.compairr_params,
                                                                               params.result_path)

        matrix_repertoire_ids = sequence_presence_matrix.columns.values

        sequence_presence_matrix = sequence_presence_matrix.to_numpy()
        sequence_presence_matrix[sequence_presence_matrix > 1] = 1

        return sequence_presence_matrix, matrix_repertoire_ids


    def _encode_data(self, dataset: RepertoireDataset, params: EncoderParams):
        label = params.label_config.get_labels_by_name()[0]

        examples = self._calculate_sequence_abundance(dataset, self.sequence_presence_matrix, self.matrix_repertoire_ids, label, params)

        encoded_data = EncodedData(examples, dataset.get_metadata([label]) if params.encode_labels else None, dataset.get_repertoire_ids(),
                                   [CompAIRRSequenceAbundanceEncoder.RELEVANT_SEQUENCE_ABUNDANCE, CompAIRRSequenceAbundanceEncoder.TOTAL_SEQUENCE_ABUNDANCE],
                                   encoding=CompAIRRSequenceAbundanceEncoder.__name__, info={'relevant_sequence_path': self.relevant_sequence_csv_path})

        encoded_dataset = RepertoireDataset(labels=dataset.labels, encoded_data=encoded_data, repertoires=dataset.repertoires)

        return encoded_dataset

    # todo untangle params
    def _calculate_sequence_abundance(self, dataset: RepertoireDataset, sequence_presence_matrix, matrix_repertoire_ids, label_str: str, params: EncoderParams):
        sequence_p_values = self._find_label_associated_sequence_p_values(sequence_presence_matrix, matrix_repertoire_ids, dataset, params, label_str)

        relevant_sequence_indices = self._get_relevant_sequence_indices(params, label_str, sequence_p_values)

        abundance_matrix = self._build_abundance_matrix(sequence_presence_matrix, matrix_repertoire_ids, dataset.get_repertoire_ids(), relevant_sequence_indices)

        return abundance_matrix


    def _prepare_compairr_input_files(self, dataset, full_sequence_set, result_path):
        if self.repertoires_filepath is None:
            self.repertoires_filepath = result_path / "compairr_repertoires.tsv"

        CompAIRRHelper.write_repertoire_file(dataset, self.repertoires_filepath, self.compairr_params)

        if self.sequences_filepath is None:
            self.sequences_filepath = result_path / "compairr_sequences.tsv"

        self.write_sequence_set_file(full_sequence_set, self.sequences_filepath)


    def _get_relevant_sequence_indices(self, params, label_str, sequence_p_values):
        if self.relevant_indices_path is None:
            self.relevant_indices_path = params.result_path / 'relevant_sequence_indices.pickle'

        if params.learn_model:
            SequenceFilterHelper._check_label_object(params, label_str)
            relevant_sequence_indices = np.array(sequence_p_values) < self.p_value_threshold

            with self.relevant_indices_path.open("wb") as file:
                pickle.dump(relevant_sequence_indices, file)

            self._write_relevant_sequence_csv(self.full_sequence_set[relevant_sequence_indices], params.result_path)

        else:
            with self.relevant_indices_path.open("rb") as file:
                relevant_sequence_indices = pickle.load(file)

        return relevant_sequence_indices

    def _write_relevant_sequence_csv(self, relevant_sequences, result_path):
        if self.relevant_sequence_csv_path is None:
            self.relevant_sequence_csv_path = result_path / 'relevant_sequences.csv'

        df = pd.DataFrame(relevant_sequences,
                          columns=self.get_relevant_sequence_attributes())

        df.to_csv(self.relevant_sequence_csv_path, sep=",", index=False)


    def _find_label_associated_sequence_p_values(self, sequence_presence_matrix, matrix_repertoire_ids, dataset, params, label_str):
        # todo make this part clearer.. maybe subset already before sending in?
        relevant = np.isin(matrix_repertoire_ids, dataset.get_repertoire_ids())
        sequence_presence_matrix = sequence_presence_matrix[:,relevant]
        matrix_repertoire_ids = matrix_repertoire_ids[relevant]

        is_first_class = self._is_first_class(dataset, matrix_repertoire_ids, params, label_str)

        sequence_p_values = []

        for sequence_vector in sequence_presence_matrix:
            if sequence_vector.sum() > 1:

                first_class_present = np.sum(sequence_vector[np.logical_and(sequence_vector, is_first_class)])
                second_class_present = np.sum(
                    sequence_vector[np.logical_and(sequence_vector, np.logical_not(is_first_class))])
                first_class_absent = np.sum(np.logical_and(is_first_class, sequence_vector == 0))
                second_class_absent = np.sum(np.logical_and(np.logical_not(is_first_class), sequence_vector == 0))

                sequence_p_values.append(fisher.pvalue(first_class_present, second_class_present, first_class_absent,
                                                       second_class_absent).right_tail)
            else:
                sequence_p_values.append(SequenceFilterHelper.INVALID_P_VALUE)

        return sequence_p_values


    def _is_first_class(self, dataset, matrix_repertoire_ids, params, label_str):
        # todo untangle
        label = params.label_config.get_label_object(label_str)
        # is_first_class = np.array([repertoire.metadata[label.name] for repertoire in repertoires]) == label.positive_class

        is_first_class = np.array(
            [dataset.get_repertoire(repertoire_identifier=repertoire_id).metadata[label.name] for repertoire_id in
             matrix_repertoire_ids]) == label.positive_class


        return is_first_class


    def get_sequence_set_for_repertoire(self, repertoire, sequence_attributes):
        return set(zip(*[value for value in repertoire.get_attributes(sequence_attributes).values() if value is not None]))

    def get_sequence_set(self, repertoire_dataset):
        attributes = self.get_relevant_sequence_attributes()
        sequence_set = set()

        for repertoire in repertoire_dataset.get_data():
            sequence_set.update(self.get_sequence_set_for_repertoire(repertoire, attributes))

        return np.array(list(sequence_set))

    def get_relevant_sequence_attributes(self):
        attributes = [EnvironmentSettings.get_sequence_type().value]

        if not self.compairr_params.ignore_genes:
            attributes += ["v_genes", "j_genes"]

        return attributes

    def write_sequence_set_file(self, sequence_set, filename):
        with open(filename, "w") as file: # todo use get_relevant_sequence_attributes
            file.write("junction_aa\tv_call\tj_call\tduplicate_count\trepertoire_id\n")

            for id, sequence_info in enumerate(sequence_set):
                file.write("\t".join(sequence_info) + f"\t1\t{id}\n")


    def _build_abundance_matrix(self, sequence_presence_matrix, matrix_repertoire_ids, dataset_repertoire_ids, sequence_p_values_indices):
        abundance_matrix = np.zeros((len(dataset_repertoire_ids), 2))

        for idx_in_dataset, dataset_repertoire_id in enumerate(dataset_repertoire_ids):
            relevant_row = np.where(matrix_repertoire_ids == dataset_repertoire_id)

            repertoire_vector = sequence_presence_matrix.T[relevant_row]

            relevant_sequence_abundance = np.sum(repertoire_vector[np.logical_and(sequence_p_values_indices, repertoire_vector)])
            total_sequence_abundance = np.sum(repertoire_vector)
            abundance_matrix[idx_in_dataset] = [relevant_sequence_abundance, total_sequence_abundance]

        return abundance_matrix


    def set_context(self, context: dict):
        self.context = context
        return self

    def store(self, encoded_dataset, params: EncoderParams):
        EncoderHelper.store(encoded_dataset, params)

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path: #todo
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        # UtilIO.export_comparison_data(encoder.comparison_data, path)
        return encoder_file

    def get_additional_files(self) -> List[Path]:
        return [self.relevant_indices_path]

    @staticmethod
    def load_encoder(encoder_file: Path): # todo
        encoder = DatasetEncoder.load_encoder(encoder_file)
        encoder.relevant_indices_path = DatasetEncoder.load_attribute(encoder, encoder_file, "relevant_indices_path")
        # encoder.comparison_data = UtilIO.import_comparison_data(encoder_file.parent)
        return encoder

    @staticmethod
    def get_documentation():
        doc = str(CompAIRRSequenceAbundanceEncoder.__doc__)

        valid_field_values = str(Repertoire.FIELDS)[1:-1].replace("'", "`")
        mapping = {
            "Valid comparison value can be any repertoire field name.": f"Valid values are {valid_field_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
