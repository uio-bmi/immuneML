import logging
from pathlib import Path

import numpy as np


from immuneML.analysis.SequenceMatcher import SequenceMatcher
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator

from immuneML.util.PathBuilder import PathBuilder


class SimilarToPositiveSequenceEncoder(DatasetEncoder):
    """
    A simple baseline encoding, to be used in combination with <todo add>
    This encoder keeps track of all positive sequences in the training set, and ignores the negative sequences.
    Any sequence within a given hamming distance from a positive training sequence will be classified positive,
    all other sequences will be classified negative.

    Arguments:

        hamming_distance (int):

        compairr_path

        ignore_genes (bool): only used when compairr is used

        threads (int): The number of threads to use for parallelization. This does not affect the results of the encoding, only the speed.
        The default number of threads is 8.

        keep_temporary_files (bool): whether to keep temporary files, including CompAIRR input, output and log files, and the sequence
        presence matrix. This may take a lot of storage space if the input dataset is large. By default temporary files are not kept.



    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_sequence_encoder:
                SimilarToPositiveSequenceEncoder:
                    hamming_distance: 2
    """

    def __init__(self, hamming_distance: int = None, compairr_path: str = None,
                 ignore_genes: bool = None, threads: int = None, keep_temporary_files: bool = None,
                 name: str = None):
        self.hamming_distance = hamming_distance
        self.compairr_path = Path(compairr_path) if compairr_path is not None else None
        self.ignore_genes = ignore_genes
        self.threads = threads
        self.keep_temporary_files = keep_temporary_files

        self.positive_sequences = None
        self.name = name
        self.context = None

    @staticmethod
    def _prepare_parameters(hamming_distance: int = None, compairr_path: str = None, ignore_genes: bool = None,
                            threads: int = None, keep_temporary_files: bool = None, name: str = None):
        location = SimilarToPositiveSequenceEncoder.__name__

        ParameterValidator.assert_type_and_value(hamming_distance, int, location, "hamming_distance", min_inclusive=0)

        if compairr_path is not None:
            ParameterValidator.assert_type_and_value(compairr_path, str, location, "compairr_path")
            CompAIRRHelper.check_compairr_path(compairr_path)

            ParameterValidator.assert_type_and_value(ignore_genes, bool, location, "ignore_genes")
            ParameterValidator.assert_type_and_value(threads, int, location, "threads")
            ParameterValidator.assert_type_and_value(keep_temporary_files, int, location, "keep_temporary_files")


        return {
            "hamming_distance": hamming_distance,
            "ignore_genes": ignore_genes,
            "threads": threads,
            "keep_temporary_files": keep_temporary_files,
            "compairr_path": compairr_path,
            "name": name,
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, SequenceDataset):
            prepared_params = SimilarToPositiveSequenceEncoder._prepare_parameters(**params)
            return SimilarToPositiveSequenceEncoder(**prepared_params)
        else:
            raise ValueError(f"{SimilarToPositiveSequenceEncoder.__name__} is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams):
        if params.learn_model:
            EncoderHelper.check_positive_class_labels(params.label_config, SimilarToPositiveSequenceEncoder.__name__)

            self.positive_sequences = self._get_positive_sequences(dataset, params)

        return self._encode_data(dataset, params)

    def _get_positive_sequences(self, dataset, params):
        subset_path = PathBuilder.build(params.result_path / "positive_sequences")
        label_name = EncoderHelper.get_single_label_name_from_config(params.label_config,
                                                                    SimilarToPositiveSequenceEncoder.__name__)

        label_obj = params.label_config.get_label_object(label_name)
        classes = dataset.get_metadata([label_name])[label_name]

        subset_indices = [idx for idx in range(dataset.get_example_count()) if classes[idx] == label_obj.positive_class]

        return dataset.make_subset(subset_indices,
                                   path=subset_path,
                                   dataset_type=Dataset.SUBSAMPLED)

    def _encode_data(self, dataset, params: EncoderParams):
        examples = self.get_sequence_matching_feature(dataset, params)

        labels = EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=examples,
                                                   labels=labels,
                                                   feature_names=["similar_to_positive_sequence"],
                                                   feature_annotations=None,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=SimilarToPositiveSequenceEncoder.__name__,
                                                   example_weights=dataset.get_example_weights(),
                                                   info={})

        return encoded_dataset

    def get_sequence_matching_feature(self, dataset, params: EncoderParams):
        if self.compairr_path is None:
            return self.get_sequence_matching_feature_without_compairr(dataset)
        else:
            return self.get_sequence_matching_feature_with_compairr(dataset, params)

    def get_sequence_matching_feature_with_compairr(self, dataset, params: EncoderParams):
        import subprocess
        import shutil

        compairr_result_path = PathBuilder.build(params.result_path / "compairr_data")
        pos_sequences_path = compairr_result_path / "positive_sequences.tsv"
        all_sequences_path = compairr_result_path / "all_sequences.tsv"

        compairr_params = self._get_compairr_params()

        CompAIRRHelper.write_sequences_file(self.positive_sequences, pos_sequences_path, compairr_params, repertoire_id="positive_sequences")
        CompAIRRHelper.write_sequences_file(dataset, all_sequences_path, compairr_params, repertoire_id="all_sequences")

        args = CompAIRRHelper.get_cmd_args(compairr_params, [all_sequences_path, pos_sequences_path], compairr_result_path, mode="-x")
        logging.info(f"{SimilarToPositiveSequenceEncoder.__name__}: running CompAIRR with the following arguments: {' '.join(args)}")

        compairr_result = subprocess.run(args, capture_output=True, text=True)
        result = CompAIRRHelper.process_compairr_output_file(compairr_result, compairr_params, compairr_result_path)

        if list(result.index) != dataset.get_example_ids():
            result = result.reindex(dataset.get_example_ids())

        if not self.keep_temporary_files:
            shutil.rmtree(compairr_result_path, ignore_errors=False, onerror=None)

        return np.array([result["positive_sequences"] > 0]).T

    def _get_compairr_params(self):
        return CompAIRRParams(compairr_path=self.compairr_path,
                              keep_compairr_input=self.keep_temporary_files,
                              differences=self.hamming_distance,
                              indels=False,
                              ignore_counts=True,
                              ignore_genes=self.ignore_genes,
                              threads=self.threads,
                              output_filename="compairr_out.txt",
                              log_filename="compairr_log.txt")

    def get_sequence_matching_feature_without_compairr(self, dataset):
        matcher = SequenceMatcher()

        examples = []

        for sequence in dataset.get_data():
            is_matching = False

            for ref_sequence in self.positive_sequences.get_data():
                if matcher.matches_sequence(sequence, ref_sequence, self.hamming_distance):
                    is_matching = True
                    break

            examples.append(is_matching)

        return np.array([examples]).T

    def _get_y_true(self, dataset, label_config: LabelConfiguration):
        labels = EncoderHelper.encode_element_dataset_labels(dataset, label_config)

        label_name = EncoderHelper.get_single_label_name_from_config(label_config, SimilarToPositiveSequenceEncoder.__name__)
        label = label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in labels[label_name]])

    def _get_positive_class(self, label_config):
        label_name = EncoderHelper.get_single_label_name_from_config(label_config, SimilarToPositiveSequenceEncoder.__name__)
        label = label_config.get_label_object(label_name)

        return label.positive_class

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        return encoder
