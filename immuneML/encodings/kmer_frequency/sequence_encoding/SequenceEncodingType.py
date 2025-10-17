from enum import Enum


class SequenceEncodingType(Enum):
    GAPPED_KMER = "GappedKmerSequenceEncoder"
    IMGT_CONTINUOUS_KMER = "IMGTKmerSequenceEncoder"
    CONTINUOUS_KMER = "KmerSequenceEncoder"
    IMGT_GAPPED_KMER = "IMGTGappedKmerEncoder"
    V_GENE_CONT_KMER = "VGeneContKmerEncoder"
    V_GENE_IMGT_KMER = 'VGeneIMGTKmerEncoder'
