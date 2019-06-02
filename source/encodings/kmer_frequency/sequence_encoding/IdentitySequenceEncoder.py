import warnings

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.environment.Constants import Constants


class IdentitySequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams):
        """
        Encodes a ReceptorSequence based on information from within the ReceptorSequence and SequenceMetadata
        instances. This allows for looking at frequency for whole sequences, with flexible definition of what a unique
        whole sequence is.
        :param sequence: ReceptorSequence
        :param params: EncoderParams (params["model"]["sequence"] and params["model"]["metadata_fields_to_include"] are
                        used)
        :return: list with only single feature
        """
        if sequence.metadata is not None and sequence.metadata.frame_type.upper() != SequenceFrameType.IN.name:
            warnings.warn('Sequence either has out or stop codon. Ignoring sequence.')
            return None

        res = []
        if params["model"].get("sequence", True):
            res.append(sequence.get_sequence())

        for field in params["model"].get("metadata_fields_to_include", []):
            if sequence.metadata is None:
                res.append("unknown")
            else:
                res.append(getattr(sequence.metadata, field))

        return [Constants.FEATURE_DELIMITER.join(res)]

    @staticmethod
    def get_feature_names(params: EncoderParams):
        res = []
        if params["model"].get("sequence", True):
            res.append("sequence")
        for field in params["model"].get("metadata_fields_to_include", []):
            res.append(field)
        return res
