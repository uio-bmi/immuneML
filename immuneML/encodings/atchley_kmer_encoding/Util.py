from collections import Counter

import numpy as np
import pandas as pd

from immuneML.encodings.atchley_kmer_encoding.RelativeAbundanceType import RelativeAbundanceType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.KmerHelper import KmerHelper


class Util:
    ATCHLEY_FACTOR_COUNT = 5
    ATCHLEY_FACTORS = None

    @staticmethod
    def compute_abundance(sequences: np.ndarray, counts: np.ndarray, k: int, abundance: RelativeAbundanceType):
        func = getattr(Util, f"compute_{abundance.value}")
        return func(sequences, counts, k)

    @staticmethod
    def compute_relative_abundance(sequences: np.ndarray, counts: np.ndarray, k: int) -> dict:
        """
        Computes the relative abundance of k-mers in the repertoire per following equations where C is the template count, T is the total count and
        RA is relative abundance (the output of the function for each k-mer separately):

        .. math::

            C^{kmer}=\\sum_{\\underset{with kmer}{TCR \\beta}} C^{TCR \\beta}

            T^{kmer} = \\sum_{kmer} C^{kmer}

            RA = \\frac{C^{kmer}}{T^{kmer}}

        For more details, please see the original publication: Ostmeyer J, Christley S, Toby IT, Cowell LG. Biophysicochemical motifs in T cell
        receptor sequences distinguish repertoires from tumor-infiltrating lymphocytes and adjacent healthy tissue. Cancer Res. Published online
        January 1, 2019:canres.2292.2018. `doi:10.1158/0008-5472.CAN-18-2292 <https://cancerres.aacrjournals.org/content/canres/79/7/1671.full.pdf>`_

        Arguments:

            sequences: an array of (amino acid) sequences (corresponding to a repertoire)
            counts: an array of counts for each of the sequences
            k: the length of the k-mer (in the publication referenced above, k is 4)

        Returns:

            a dictionary where keys are k-mers and values are their relative abundances in the given list of sequences

        """

        c_kmers = Counter()
        for index, sequence in enumerate(sequences):
            kmers = KmerHelper.create_kmers_from_string(sequence, k)
            c_kmers += {kmer: counts[index] for kmer in kmers}

        t_kmers = sum(c_kmers.values())

        return {kmer: c_kmers[kmer] / t_kmers for kmer in c_kmers.keys()}

    @staticmethod
    def compute_tcrb_relative_abundance(sequences: np.ndarray, counts: np.ndarray, k: int) -> dict:
        """
        Computes the relative abundance of k-mers in the repertoire per following equations where C is the template count for the given receptor
        sequence, T is the total count across all receptor sequences. The relative abundance per receptor sequence is then computed and only the
        maximum sequence abudance was used for the k-mer so that the k-mer's relative abundance is equal to the abundance of the most frequent
        receptor sequence in which the receptor appears:

        .. math::

            T^{TCR \\beta} = \\sum_{TCR\\beta} C^{TCR\\beta}

            RA^{TCR\\beta} = \\frac{C^{TCR\\beta}}{T^{TCR\\beta}}

            RA = \\max_{\\underset{with \\, kmer}{TCR\\beta}} {RA^{TCR \\beta}}

        For more details, please see the original publication: Ostmeyer J, Christley S, Toby IT, Cowell LG. Biophysicochemical motifs in T cell
        receptor sequences distinguish repertoires from tumor-infiltrating lymphocytes and adjacent healthy tissue. Cancer Res. Published online
        January 1, 2019:canres.2292.2018. `doi:10.1158/0008-5472.CAN-18-2292 <https://cancerres.aacrjournals.org/content/canres/79/7/1671.full.pdf>`_

        Arguments:

            sequences: an array of (amino acid) sequences (corresponding to a repertoire)
            counts: an array of counts for each of the sequences
            k: the length of the k-mer (in the publication referenced above, k is 4)

        Returns:

            a dictionary where keys are k-mers and values are their relative abundances in the given list of sequences

        """
        relative_abundance = {}
        total_count = np.sum(counts)
        relative_abundance_per_sequence = counts / total_count
        for index, sequence in enumerate(sequences):
            kmers = KmerHelper.create_kmers_from_string(sequence, k)
            for kmer in kmers:
                if kmer not in relative_abundance or relative_abundance[kmer] < relative_abundance_per_sequence[index]:
                    relative_abundance[kmer] = relative_abundance_per_sequence[index]

        return relative_abundance

    @staticmethod
    def get_atchely_factors(kmers: list, k: int) -> pd.DataFrame:
        """
        Returns values of Atchley factors for each amino acid in the sequence. The data was downloaded from the publication:
        Atchley WR, Zhao J, Fernandes AD, Drüke T. Solving the protein sequence metric problem. PNAS. 2005;102(18):6395-6400.
        `doi:10.1073/pnas.0408677102 <https://www.pnas.org/content/102/18/6395>`_

        Arguments:

            kmers: a list of amino acid sequences
            k: length of k-mers

        Returns:

            values of Atchley factors for each amino acid in the sequence

        """
        if Util.ATCHLEY_FACTORS is None:
            Util.ATCHLEY_FACTORS = pd.read_csv(EnvironmentSettings.root_path / "immuneML/encodings/atchley_kmer_encoding/atchley_factors.csv",
                                               index_col='amino_acid')

        factors = np.zeros((len(kmers), Util.ATCHLEY_FACTOR_COUNT * k))
        for index, kmer in enumerate(kmers):
            factors[index] = np.concatenate([Util.ATCHLEY_FACTORS.loc[amino_acid].values for amino_acid in kmer])

        factors_df = pd.DataFrame(factors, index=kmers)

        return factors_df
