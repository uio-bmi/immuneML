import dataclasses
from typing import Protocol
from scipy.stats import poisson
import numpy as np
from bionumpy import EncodedRaggedArray, EncodedArray, count_encoded

from immuneML.ml_methods.generative_models.SequenceTransitionDistribution import EncodedLookup


@dataclasses.dataclass
class Poisson:
    mu: float

    def log_prob(self, x):
        return poisson.logpmf(x, self.mu)

    def sample(self, n_samples):
        return poisson.rvs(self.mu, size=n_samples)


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


class SmoothedLengthDistribution:
    def __init__(self, empirical_distribution, smooth_distribution, p_smooth):
        self.empirical_distribution = empirical_distribution
        self.smooth_distribution = smooth_distribution
        self.p_smooth = p_smooth

    def sample(self, count: int) -> np.ndarray:
        is_smooth = np.random.rand(count) < self.p_smooth
        smooth_samples = self.smooth_distribution.sample(count)
        empirical_samples = self.empirical_distribution.sample(count)
        return np.where(is_smooth, smooth_samples, empirical_samples)

    def log_prob(self, lengths: np.ndarray) -> np.ndarray:
        return np.logaddexp(np.log(1 - self.p_smooth) + self.empirical_distribution.log_prob(lengths),
                            np.log(self.p_smooth) + self.smooth_distribution.log_prob(lengths))


class KmerModel:
    def __init__(self, kmer_probs: EncodedLookup):
        self.kmer_probs = kmer_probs
        self._raw_values = np.arange(kmer_probs.alphabet_size, dtype=np.uint8)
        self._log_probs = np.log(kmer_probs)

    def sample(self, count: int) -> EncodedRaggedArray:
        kmer_hashes = np.random.choice(self._raw_values, size=count, p=self.kmer_probs.raw())
        return EncodedArray(kmer_hashes, self.kmer_probs.encoding)

    def log_prob(self, kmers: EncodedRaggedArray) -> np.ndarray:
        return self._log_probs[kmers]


class MultinomialKmerModel:
    def __init__(self, kmer_probs: EncodedLookup, sequence_length: int):
        self.kmer_model = KmerModel(kmer_probs)
        self.sequence_length = sequence_length

    def sample(self, count: int) -> EncodedRaggedArray:
        if hasattr(self.sequence_length, 'sample'):
            sequence_length = self.sequence_length.sample(count)
        else:
            sequence_length = np.full((count,), self.sequence_length, dtype=int)
        total_kmers = sequence_length.sum()
        kmers = self.kmer_model.sample(total_kmers)
        # kmer_hashes = np.random.choice(self._raw_values, size=total_kmers,
        # p=self.kmer_probs.raw())
        # kmers = EncodedArray(kmer_hashes, self.kmer_probs.encoding)
        return EncodedRaggedArray(kmers, sequence_length)

    def log_prob(self, kmers: EncodedRaggedArray) -> np.ndarray:
        lengths = kmers.shape[-1]
        if hasattr(self.sequence_length, 'log_prob'):
            length_log_probs = self.sequence_length.log_prob(lengths)
        else:
            assert np.all(lengths == self.sequence_length)
            length_log_probs = 0
        return self.kmer_model.log_prob(kmers).sum(axis=-1) + length_log_probs


def estimate_length_distribution(lengths: np.ndarray) -> EmpiricalLengthDistribution:
    counts = np.bincount(lengths)
    return EmpiricalLengthDistribution(counts / counts.sum())


def estimate_smoothed_length_distribution(lengths: np.ndarray, prior_count=1) -> SmoothedLengthDistribution:
    empirical = estimate_length_distribution(lengths)
    smooth = Poisson(np.mean(lengths))
    return SmoothedLengthDistribution(empirical, smooth, prior_count / (prior_count + len(lengths)))


def estimate_kmer_model(kmers: EncodedRaggedArray, prior_count=1) -> MultinomialKmerModel:
    length_distribution = estimate_smoothed_length_distribution(kmers.lengths, prior_count=prior_count)
    kmer_counts = count_encoded(kmers, axis=None)
    lookup = EncodedLookup((kmer_counts.counts + prior_count / len(kmer_counts.counts)) / (kmers.size + prior_count),
                           kmers.encoding)

    return MultinomialKmerModel(lookup, length_distribution)
