from typing import Protocol

import numpy as np
from bionumpy import EncodedRaggedArray, EncodedArray, count_encoded

from immuneML.ml_methods.generative_models.transition_distribution import EncodedLookup


class KmerDistribution(Protocol):
    def sample(self, count: int) -> EncodedRaggedArray:
        ...

    def log_prob(self, kmers) -> np.ndarray:
        ...


class EmpiricalLengthDistribution:
    def __init__(self, lengths_frequencies: np.ndarray):
        self.lengths_frequencies = lengths_frequencies

    def sample(self, count: int) -> np.ndarray:
        return np.random.choice(np.arange(len(self.lengths_frequencies)), size=count, p=self.lengths_frequencies)

    def log_prob(self, lengths: np.ndarray) -> np.ndarray:
        mask = lengths < len(self.lengths_frequencies)
        probs = np.zeros_like(lengths, dtype=float)
        probs[mask] = self.lengths_frequencies[lengths[mask]]
        return np.log(probs)


class MultinomialKmerModel:
    def __init__(self, kmer_probs: EncodedLookup, sequence_length: int):
        self.kmer_probs = kmer_probs
        self.sequence_length = sequence_length
        self._raw_values = np.arange(kmer_probs.alphabet_size, dtype=np.uint8)
        self._log_probs = np.log(kmer_probs)

    def sample(self, count: int) -> EncodedRaggedArray:
        if hasattr(self.sequence_length, 'sample'):
            sequence_length = self.sequence_length.sample(count)
        else:
            sequence_length = np.full((count,), self.sequence_length, dtype=int)
        total_kmers = sequence_length.sum()
        kmer_hashes = np.random.choice(self._raw_values, size=total_kmers,
                                       p=self.kmer_probs.raw())
        kmers = EncodedArray(kmer_hashes, self.kmer_probs.encoding)
        return EncodedRaggedArray(kmers, sequence_length)

    def log_prob(self, kmers: EncodedRaggedArray) -> np.ndarray:
        lengths = kmers.shape[-1]
        if hasattr(self.sequence_length, 'log_prob'):
            length_log_probs = self.sequence_length.log_prob(lengths)
        else:
            assert np.all(lengths == self.sequence_length)
            length_log_probs = 0
        return self._log_probs[kmers].sum(axis=-1) + length_log_probs


def estimate_length_distribution(lengths: np.ndarray) -> EmpiricalLengthDistribution:
    counts = np.bincount(lengths)
    return EmpiricalLengthDistribution(counts / counts.sum())


def estimate_kmer_model(kmers: EncodedRaggedArray) -> MultinomialKmerModel:
    length_distribution = estimate_length_distribution(kmers.lengths)
    kmer_counts = count_encoded(kmers, axis=None)
    lookup = EncodedLookup(kmer_counts.counts / kmers.size, kmers.encoding)
    return MultinomialKmerModel(lookup, length_distribution)
