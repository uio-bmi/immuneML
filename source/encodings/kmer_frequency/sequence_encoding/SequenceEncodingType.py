from enum import Enum


class SequenceEncodingType(Enum):
    GAPPED_KMER = "source.encodings.kmer_frequency.sequence_encoding.GappedKmerSequenceEncoder.GappedKmerSequenceEncoder"
    IMGT_CONTINUOUS_KMER = "source.encodings.kmer_frequency.sequence_encoding.IMGTKmerSequenceEncoder.IMGTKmerSequenceEncoder"
    CONTINUOUS_KMER = "source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder.KmerSequenceEncoder"
    IDENTITY = "source.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder.IdentitySequenceEncoder"
    IMGT_GAPPPED_KMER = "source.encodings.kmer_frequency.sequence_encoding.IMGTGappedKmerEncoder.IMGTGappedKmerEncoder"
