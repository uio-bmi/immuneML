import warnings

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingResult import SequenceEncodingResult
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.environment.Constants import Constants


class IdentitySequenceEncoder(SequenceEncodingStrategy):

    @staticmethod
    def encode_sequence(sequence: ReceptorSequence, params: EncoderParams) -> SequenceEncodingResult:
        """
        Encodes a ReceptorSequence based on information from within the ReceptorSequence and SequenceMetadata
        instances. This allows for looking at frequency for whole sequences, with flexible definition of what a unique
        whole sequence is.
        :param sequence: ReceptorSequence
        :param params: EncoderParams (params["model"]["sequence"] and params["model"]["metadata_fields_to_include"] are
                        used)
        :return: SequenceEncodingResult
        """

        if sequence.metadata is not None and sequence.metadata.frame_type.upper() in [SequenceFrameType.OUT.name, SequenceFrameType.STOP.name]:
            warnings.warn('Sequence either has out or stop codon. Ignoring sequence.')
            return SequenceEncodingResult(None, None)

        res = {}

        if params["model"].get("sequence", True):
            res["sequence"] = sequence.get_sequence()

        for field in params["model"].get("metadata_fields_to_include", []):
            if sequence.metadata is None:
                res[field] = Constants.UNKNOWN
            else:
                res[field] = getattr(sequence.metadata, field)

        return SequenceEncodingResult(features=[Constants.FEATURE_DELIMITER.join(res.values())],
                                      feature_information_names=list(res.keys()))
