from importlib import import_module

from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType


class EncodingParser:

    @staticmethod
    def parse_encoder(workflow_specification: dict):
        if workflow_specification["encoder"] == "KmerFrequencyEncoder":
            assert "sequence_encoding" in workflow_specification[
                "encoder_params"], "Parser: creating encoder: sequence_encoding for KmerFrequencyEncoder is not specified."
            encoder = KmerFrequencyEncoder()
        else:
            assert "model" in workflow_specification["encoder_params"] and "model_creator" in \
                   workflow_specification["encoder_params"][
                       "model"], "Parser: creating encoder: model_creator for Word2VecEncoder is not specified."
            encoder = Word2VecEncoder()

        encoder_params = EncodingParser._parse_encoder_params(workflow_specification["encoder_params"], encoder)

        return encoder, encoder_params

    @staticmethod
    def _parse_encoder_params(encoder_params: dict, encoder: DatasetEncoder) -> dict:
        parsed_encoder_params = {}

        if isinstance(encoder, KmerFrequencyEncoder):
            parsed_encoder_params["sequence_encoding_strategy"] = EncodingParser._transform_sequence_encoding_strategy(
                encoder_params["sequence_encoding"])
            parsed_encoder_params["reads"] = ReadsType.UNIQUE if encoder_params["reads"] == "unique" else ReadsType.ALL
            parsed_encoder_params["normalization_type"] = NormalizationType.L2 if encoder_params[
                                                                                      "normalization_type"] == "l2" else NormalizationType.RELATIVE_FREQUENCY
        elif isinstance(encoder, Word2VecEncoder):
            parsed_encoder_params["model"] = {
                "k": encoder_params["model"]["k"],
                "size": encoder_params["model"]["size"],
                "model_creator": ModelType.SEQUENCE if encoder_params["model"][
                                                           "model_creator"] == "receptor_sequence" else ModelType.KMER_PAIR
            }

        for key in encoder_params.keys():
            if key not in parsed_encoder_params.keys():
                parsed_encoder_params[key] = encoder_params[key]

        return parsed_encoder_params

    @staticmethod
    def _transform_sequence_encoding_strategy(sequence_encoding_strategy: str) -> SequenceEncodingStrategy:
        val = getattr(SequenceEncodingType, sequence_encoding_strategy.upper()).value
        (module_path, _, class_name) = val.rpartition(".")
        module = import_module(module_path)
        sequence_encoding_strategy_instance = getattr(module, class_name)()
        return sequence_encoding_strategy_instance
