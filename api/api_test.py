from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFreqRepertoireEncoder import KmerFreqRepertoireEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.Label import Label
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.ReflectionHandler import ReflectionHandler


def encode_repertoire_by_kmer_freq(repertoire_file_path: str, result_path: str, repertoire_format: str = "AdaptiveBiotech", k: int = 3,
                                   label_name: str = "CMV", label_value=False):
    repertoire_loader = ReflectionHandler.get_class_by_name(repertoire_format, "dataset_import/")
    repertoire = repertoire_loader.load_repertoire_from_file(repertoire_file_path, {"result_path": result_path,
                                                                                    "metadata": {label_name: label_value}})

    encoder = KmerFreqRepertoireEncoder(normalization_type=NormalizationType.RELATIVE_FREQUENCY,
                                        reads=ReadsType.UNIQUE,
                                        sequence_encoding=SequenceEncodingType.CONTINUOUS_KMER,
                                        k=3)

    counts, _, _, _ = encoder.encode_repertoire(repertoire, EncoderParams(result_path=result_path,
                                                                          label_configuration=LabelConfiguration([Label(label_name)])))
    return counts
