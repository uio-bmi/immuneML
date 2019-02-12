from enum import Enum


class SequenceEncodingType(Enum):
    GAPPED_KMER = 1
    IMGT_CONTINUOUS_KMER = 2
    CONTINUOUS_KMER = 3
    IDENTITY = 4
    IMGT_GAPPPED_KMER = 5
