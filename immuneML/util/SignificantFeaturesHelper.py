from pathlib import Path
import numpy as np
import os
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import CompAIRRSequenceAbundanceEncoder
from immuneML.encodings.abundance_encoding.KmerAbundanceEncoder import KmerAbundanceEncoder
from immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ParameterValidator import ParameterValidator


class SignificantFeaturesHelper:

    @staticmethod
    def parse_parameters(kwargs, location):
        ParameterValidator.assert_keys_present(kwargs.keys(), ["p_values", "k_values", "label"], location, location)

        ParameterValidator.assert_type_and_value(kwargs["p_values"], list, location, "p_values")
        ParameterValidator.assert_type_and_value(kwargs["k_values"], list, location, "k_values")

        assert isinstance(kwargs["label"], dict), f"{location}: {kwargs['label']} is not a valid value for parameter label. " \
                                                  f"It has to be of type dict, but is now of type {type(kwargs['label']).__name__}." \
                                                  f"Did you remember to set the positive_class?"

        assert len(kwargs["label"]) == 1, f"{location}: only one label is allowed to be set, found {len(kwargs['label'])}: {list(kwargs['label'])}"

        label_name = list(kwargs["label"].keys())[0]

        assert "positive_class" in kwargs["label"][label_name], f"{location}: positive_class must be set for label {label_name}"

        assert len(kwargs["p_values"]) == len(set( kwargs["p_values"])), f"{location}: p_values should only contain unique values, found {kwargs['p_values']}"
        assert len(kwargs["k_values"]) == len(set( kwargs["k_values"])), f"{location}: k_values should only contain unique values, found {kwargs['k_values']}"

        ParameterValidator.assert_all_type_and_value(kwargs["p_values"], float, "location", "p_values", min_inclusive=0)

        for value in kwargs["k_values"]:
            if value != "full_sequence":
                ParameterValidator.assert_type_and_value(value, int, location, "k_values", 1)

        if "compairr_path" in kwargs and kwargs["compairr_path"] is not None:
            ParameterValidator.assert_type_and_value(kwargs["compairr_path"], str, location, "compairr_path")
            kwargs["compairr_path"] = Path(kwargs["compairr_path"])

        return kwargs

    @staticmethod
    def parse_sequences_path(kwargs, field_name, location):
        ParameterValidator.assert_keys_present(kwargs.keys(), [field_name], location, location)
        ParameterValidator.assert_type_and_value(kwargs[field_name], str, location,
                                                 field_name)
        assert os.path.isfile(kwargs[field_name]), f"{location}: implanted_sequences_path does not exist: {kwargs['field_name']}"

        kwargs[field_name] = Path(kwargs[field_name])

        return kwargs

    @staticmethod
    def load_sequences(groundtruth_sequences_path, trim_leading_trailing=False):
        with open(groundtruth_sequences_path) as f:
            readlines = f.readlines()
            if trim_leading_trailing:
                sequences = [seq.strip()[1:-1] for seq in readlines]
            else:
                sequences = [seq.strip() for seq in readlines]
        return sequences

    @staticmethod
    def _get_encoder_name(k):
        encoder_name = f"{k}-mer" if type(k) == int else k
        return encoder_name

    @staticmethod
    def _build_encoder_params(label_config, encoder_result_path):
        encoder_params = EncoderParams(result_path=encoder_result_path,
                                       label_config=label_config,
                                       pool_size=1,
                                       learn_model=True,
                                       encode_labels=False)

        return encoder_params

    @staticmethod
    def _build_kmer_encoder(dataset, k, p_value, encoder_params):
        encoder = KmerAbundanceEncoder(p_value_threshold=p_value,
                                       sequence_encoding=SequenceEncodingType.CONTINUOUS_KMER,
                                       k=k, k_left=0, k_right=0, min_gap=0, max_gap=0)

        encoder.encode(dataset, encoder_params)

        return encoder

    @staticmethod
    def _build_sequence_encoder(dataset, p_value, encoder_params):
        encoder = SequenceAbundanceEncoder(comparison_attributes=[EnvironmentSettings.get_sequence_type().value],
                                           p_value_threshold=p_value, sequence_batch_size=100000, repertoire_batch_size=16)

        encoder.encode(dataset, encoder_params)

        return encoder

    @staticmethod
    def _build_compairr_sequence_encoder(dataset, p_value, encoder_params, compairr_path):
        encoder = CompAIRRSequenceAbundanceEncoder(p_value_threshold=p_value, compairr_path=compairr_path,
                                                   sequence_batch_size=100000, ignore_genes=True, threads=8,
                                                   keep_temporary_files=True)

        encoder.encode(dataset, encoder_params)

        return encoder

    @staticmethod
    def _get_relevant_feature_presence(encoder, relevant_indices):

        if isinstance(encoder, KmerAbundanceEncoder):
            relevant_feature_presence = np.sum(encoder.kmer_presence_matrix[relevant_indices], axis=0)
        elif isinstance(encoder, CompAIRRSequenceAbundanceEncoder):
            relevant_feature_presence = np.sum(encoder.sequence_presence_matrix[relevant_indices], axis=0)
        else:
            relevant_feature_presence = np.zeros(shape=(6,))

            for i, sequence_vector in enumerate(encoder.comparison_data.get_item_vectors()):
                if relevant_indices[i]:
                    relevant_feature_presence += sequence_vector

        return relevant_feature_presence
