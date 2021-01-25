from enum import Enum


class SequenceEncodingType(Enum):
    GAPPED_KMER = "GappedKmerSequenceEncoder"
    IMGT_CONTINUOUS_KMER = "IMGTKmerSequenceEncoder"
    CONTINUOUS_KMER = "KmerSequenceEncoder"
    IDENTITY = "IdentitySequenceEncoder"
    IMGT_GAPPED_KMER = "IMGTGappedKmerEncoder"
