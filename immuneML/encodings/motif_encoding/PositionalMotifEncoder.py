import logging
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

from immuneML.caching.CacheHandler import CacheHandler
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.PositionalMotifParams import PositionalMotifParams
from immuneML.encodings.motif_encoding.WeightedSequenceContainer import WeigthedSequenceContainer
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator


from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.encodings.motif_encoding.WeightHelper import WeightHelper


class PositionalMotifEncoder(DatasetEncoder):
    """
    xxx
    todo docs

    can only be used for sequences of the same length

    Arguments:

        max_positions (int):

        min_precision (float):

        min_recall (float):

        min_recall_before_merging (float):

        min_true_positives (int):

        generalize_motifs (bool):

        candidate_motif_filepath (str):

        label (str):

        use_weights (bool):

        # todo should weighting be a parameter here?





    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_motif_encoder:
                MotifEncoder:
                    ...


    """

    dataset_mapping = {
        "SequenceDataset": "PositionalMotifSequenceEncoder",
    }

    def __init__(self, max_positions: int = None, min_precision: float = None, min_recall: float = None,
                 min_recall_before_merging: float = None, min_true_positives: int = None, generalize_motifs: bool = False,
                 use_weights: bool = True, candidate_motif_filepath: str = None, label: str = None,
                 name: str = None):
        self.max_positions = max_positions
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_recall_before_merging = min_recall_before_merging
        self.min_true_positives = min_true_positives
        self.generalize_motifs = generalize_motifs
        self.use_weights = use_weights
        self.candidate_motif_filepath = candidate_motif_filepath
        self.significant_motif_filepath = None

        self.label = label
        self.name = name
        self.context = None

    @staticmethod
    def _prepare_parameters(max_positions: int = None, min_precision: float = None, min_recall: float = None,
                            min_recall_before_merging: float = None, min_true_positives: int = None,
                            generalize_motifs: bool = False, use_weights: bool = False, candidate_motif_filepath: str = None,
                            label: str = None, name: str = None):

        location = PositionalMotifEncoder.__name__

        ParameterValidator.assert_type_and_value(max_positions, int, location, "max_positions", min_inclusive=1)
        ParameterValidator.assert_type_and_value(min_precision, (int, float), location, "min_precision", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_recall, (int, float), location, "min_recall", min_inclusive=0, max_inclusive=1)
        ParameterValidator.assert_type_and_value(min_true_positives, int, location, "min_true_positives", min_inclusive=1)
        ParameterValidator.assert_type_and_value(generalize_motifs, bool, location, "generalize_motifs")
        ParameterValidator.assert_type_and_value(use_weights, bool, location, "use_weights")

        if min_recall_before_merging is None:
            min_recall_before_merging = min_recall
        else:
            ParameterValidator.assert_type_and_value(min_recall_before_merging, (int, float), location, "min_recall_before_merging", min_inclusive=0, max_inclusive=1)

        assert min_recall >= min_recall_before_merging, f"{location}: min_recall_before_merging (value = {min_recall_before_merging}) cannot exceed min_recall (value = {min_recall})"

        if candidate_motif_filepath is not None:
            ParameterValidator.assert_type_and_value(candidate_motif_filepath, str, location, "candidate_motif_filepath")

            candidate_motif_filepath = Path(candidate_motif_filepath)
            assert candidate_motif_filepath.is_file(), f"{location}: the file {candidate_motif_filepath} does not exist. " \
                                                       f"Specify the correct path under motif_filepath."

            file_columns = list(pd.read_csv(candidate_motif_filepath, sep="\t", iterator=False, dtype=str, nrows=0).columns)

            ParameterValidator.assert_all_in_valid_list(file_columns,
                                                        ["indices", "amino_acids"],
                                                        location, "candidate_motif_filepath (column names)")

        if label is not None:
            ParameterValidator.assert_type_and_value(label, str, location, "label")


        return {
            "max_positions": max_positions,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_recall_before_merging": min_recall_before_merging,
            "min_true_positives": min_true_positives,
            "generalize_motifs": generalize_motifs,
            "use_weights": use_weights,
            "label": label,
            "name": name,
        }

    @staticmethod
    def build_object(dataset=None, **params):
        if isinstance(dataset, SequenceDataset):
            prepared_params = PositionalMotifEncoder._prepare_parameters(**params)
            return PositionalMotifEncoder(**prepared_params)
        else:
            raise ValueError(f"{PositionalMotifEncoder.__name__} is not defined for dataset types which are not SequenceDataset.")

    def encode(self, dataset, params: EncoderParams):
        EncoderHelper.check_positive_class_labels(params.label_config, PositionalMotifEncoder.__name__)

        return self._encode_data(dataset, params)

    def _encode_data(self, dataset, params: EncoderParams):
        motifs = self._prepare_candidate_motifs(dataset, params)

        # todo after weights have been implemented properly, move stuff to get sequence_container to separate function

        labels = EncoderHelper.encode_element_dataset_labels(dataset, params.label_config)
        y_true = self._get_y_true(labels, params.label_config)
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(dataset)

        if self.use_weights:
            self.weight_pseudocount = 1  # todo these parameters need to be set by the user, probably when changing the YAML specification to specify weights
            self.weight_upper_limit = 1
            self.weight_lower_limit = 0

            positional_weights = self._get_positional_weights(np_sequences)
            weights = self._get_sequence_weights(np_sequences, positional_weights)
        else:
            print("Not using weights")
            positional_weights = None
            weights = None

        sequence_container = WeigthedSequenceContainer(np_sequences, weights, y_true)
        # todo current solution only filters motifs based on 'min recall before merging'; find a way to reduce the unnecessary extra tests (precision for generalized motifs) & maybe completely remove this option of one recall before and one after merging?
        # current solution only good if min_recall_before_merging == min_recall

        motifs = self._filter_motifs(motifs, sequence_container, params.pool_size,
                                     min_recall=self.min_recall_before_merging, generalized=False)


        if self.generalize_motifs:
            generalized_motifs = PositionalMotifHelper.get_generalized_motifs(motifs)
            generalized_motifs = self._filter_motifs(generalized_motifs, sequence_container, params.pool_size,
                                                     min_recall=self.min_recall, generalized=True)

            motifs = self._filter_motifs(motifs, sequence_container, params.pool_size,
                                                     min_recall=self.min_recall, generalized=False)

            motifs += generalized_motifs

        # todo separate file for significant motifs and generalized motifs???

        self.significant_motif_filepath = params.result_path / "significant_motifs.tsv"
        PositionalMotifHelper.write_motifs_to_file(motifs, self.significant_motif_filepath)

        examples, feature_names = self._construct_encoded_data_matrix(motifs, np_sequences)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=examples,
                                                   labels=labels,
                                                   feature_names=feature_names,
                                                   example_ids=dataset.get_example_ids(),
                                                   encoding=PositionalMotifEncoder.__name__,
                                                   info={"positional_weights": positional_weights,
                                                         "candidate_motif_filepath": self.candidate_motif_filepath,
                                                         "significant_motif_filepath": self.significant_motif_filepath})

        return encoded_dataset

    def _prepare_candidate_motifs(self, dataset, params):
        full_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        candidate_motifs = self._get_candidate_motifs(full_dataset, params.pool_size)
        assert len(candidate_motifs) > 0, f"{PositionalMotifEncoder.__name__}: no candidate motifs were found. " \
                                          f"Please try decreasing the value for parameter 'min_true_positives'."

        self.candidate_motif_filepath = params.result_path / "all_candidate_motifs.tsv"
        PositionalMotifHelper.write_motifs_to_file(candidate_motifs, self.candidate_motif_filepath)

        return candidate_motifs

    def _get_candidate_motifs(self, full_dataset, pool_size=4):
        '''Returns all candidate motifs, which are either read from the input file or computed by finding
        all motifs occuring in at least a given number of sequences of the full dataset.'''
        if self.candidate_motif_filepath is None:
            return CacheHandler.memo_by_params(self._build_candidate_motifs_params(full_dataset),
                                               lambda: self._compute_candidate_motifs(full_dataset, pool_size))
        else:
            return PositionalMotifHelper.read_motifs_from_file(self.candidate_motif_filepath)

    def _build_candidate_motifs_params(self, dataset: SequenceDataset):
        return (("dataset_identifier", dataset.identifier),
                ("sequence_ids", tuple(dataset.get_example_ids()),
                ("max_positions", self.max_positions),
                ("min_true_positives", self.min_true_positives)))

    def _compute_candidate_motifs(self, full_dataset, pool_size=4):
        np_sequences = PositionalMotifHelper.get_numpy_sequence_representation(full_dataset)
        params = PositionalMotifParams(max_positions=self.max_positions, count_threshold=self.min_true_positives,
                                       pool_size=pool_size)
        return PositionalMotifHelper.compute_all_candidate_motifs(np_sequences, params)

    def _get_y_true(self, labels, label_config: LabelConfiguration):
        label_name = self._get_label_name(label_config)
        label = label_config.get_label_object(label_name)

        return np.array([cls == label.positive_class for cls in labels[label_name]])

    def _get_label_name(self, label_config: LabelConfiguration):
        if self.label is not None:
            assert self.label in label_config.get_labels_by_name(), f"{PositionalMotifEncoder.__name__}: specified label " \
                                                                    f"'{self.label}' was not present among the dataset labels: " \
                                                                    f"{', '.join(label_config.get_labels_by_name())}"
            label_name = self.label
        else:
            assert label_config.get_label_count() != 0, f"{PositionalMotifEncoder.__name__}: the dataset does not contain labels, please specify a label under 'instructions'."
            assert label_config.get_label_count() == 1, f"{PositionalMotifEncoder.__name__}: multiple labels were found: {', '.join(label_config.get_labels_by_name())}. " \
                                                        f"Please reduce the number of labels to one, or use the parameter 'label' to specify one of these labels. "

            label_name = label_config.get_labels_by_name()[0]

        return label_name

    def _get_positional_weights(self, np_sequences):
        return WeightHelper.compute_positional_aa_contributions(np_sequences, self.weight_pseudocount)

    def _get_sequence_weights(self, np_sequences, positional_weights):
        return np.apply_along_axis(WeightHelper.compute_sequence_weight, 1, np_sequences, positional_weights=positional_weights, lower_limit=self.weight_lower_limit, upper_limit=self.weight_upper_limit)

    def check_filtered_motifs(self, filtered_motifs):
        assert len(filtered_motifs) > 0, f"{PositionalMotifEncoder.__name__}: no significant motifs were found. " \
                                         f"Please try decreasing the values for parameters 'min_precision', 'min_recall' and/or 'min_recall_before_merging'."

    def _filter_motifs(self, candidate_motifs, sequence_container, pool_size, min_recall, generalized=False):
        motif_type = "generalized motifs" if generalized else "motifs"

        logging.info(f"{PositionalMotifEncoder.__name__}: filtering {len(candidate_motifs)} {motif_type} with precision >= {self.min_precision} and recall >= {min_recall}")

        with Pool(pool_size) as pool:
            partial_func = partial(self._check_motif, sequence_container=sequence_container, min_recall=min_recall)

            filtered_motifs = list(filter(None, pool.map(partial_func, candidate_motifs)))

        if not generalized:
            self.check_filtered_motifs(filtered_motifs)

        logging.info(f"{PositionalMotifEncoder.__name__}: filtering {motif_type} done, {len(filtered_motifs)} motifs left")

        return filtered_motifs

    def _check_motif(self, motif, sequence_container, min_recall):
        indices, amino_acids = motif

        pred = PositionalMotifHelper.test_motif(sequence_container.np_sequences, indices, amino_acids)
        if sum(pred) >= self.min_true_positives:
            if precision_score(y_true=sequence_container.y_true, y_pred=pred, sample_weight=sequence_container.weights) >= self.min_precision:
                if recall_score(y_true=sequence_container.y_true, y_pred=pred, sample_weight=sequence_container.weights) >= min_recall:
                    return motif

    def _construct_encoded_data_matrix(self, motifs, np_sequences):

        feature_names = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False) for indices, amino_acids in motifs]
        examples = [PositionalMotifHelper.test_motif(np_sequences, indices, amino_acids) for indices, amino_acids in motifs]

        return np.column_stack(examples), feature_names

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