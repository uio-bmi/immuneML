import dataclasses
from numbers import Number

import numpy as np
import scipy.special
from bionumpy.encoded_array import Encoding, as_encoded_array, EncodedRaggedArray, EncodedArray
from bionumpy.encodings import AlphabetEncoding, AminoAcidEncoding
from npstructures import RaggedArray, RaggedShape


class EncodedLookup(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, lookup: np.ndarray, encoding: Encoding):
        self._lookup = np.asanyarray(lookup)
        self._encoding = encoding

    def __repr__(self):
        return repr(self._lookup)[:20]

    @property
    def alphabet_size(self):
        return self._lookup.shape[-1]

    @property
    def encoding(self):
        return self._encoding

    def raw(self):
        return self._lookup

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == "__call__"
        if isinstance(inputs, tuple):
            inputs = tuple(i._lookup if isinstance(i, EncodedLookup) else i for i in inputs)
        else:
            inputs = inputs._lookup if isinstance(inputs, EncodedLookup) else inputs
        return self.__class__(ufunc(*inputs, **kwargs), self._encoding)

    def __getitem__(self, key):
        assert not isinstance(key, Number), key
        key, shape = self._translate_key(key)
        value = self._lookup[key]
        if shape:
            value = RaggedArray(value, shape[-1])
        return value

    def __setitem__(self, key, value):
        key, shape = self._translate_key(key)
        self._lookup[key] = value

    def _translate_key(self, key):
        shape = None
        if isinstance(key, tuple):
            key = tuple(as_encoded_array(i, self._encoding).raw() if (i is not Ellipsis and i is not None) else i
                        for i in key)
            shape = tuple(i.shape if isinstance(i, RaggedArray) else (1,) for i in key)
            key = tuple(i.ravel() if isinstance(i, RaggedArray) else i for i in key)
        else:
            key = as_encoded_array(key, self._encoding).raw()
            if isinstance(key, RaggedArray):
                shape = key.shape
                key = key.ravel()
        return key, shape


@dataclasses.dataclass
class SequenceTransitionDistribution:
    transition_matrix: EncodedLookup
    initial_distribution: EncodedLookup
    end_probs: EncodedLookup

    @classmethod
    def from_probabilities(cls, *args, **kwargs):
        return cls(*(np.log(arg) for arg in args), **{k: np.log(v) for k, v in kwargs.items()})

    def log_prob(self, sequence):
        sequence = as_encoded_array(sequence, self.transition_matrix.encoding)
        prev = sequence[..., :-1]
        next = sequence[..., 1:]
        transition_log_probs = self.transition_matrix[(prev, next)].sum(axis=-1)
        start_log_probs = self.initial_distribution[sequence[..., 0]]
        end_log_probs = self.end_probs[sequence[..., -1]]
        return transition_log_probs + start_log_probs + end_log_probs

    def save(self, filename):
        np.savez(filename, transition_matrix=self.transition_matrix._lookup,
                 initial_distribution=self.initial_distribution._lookup,
                 end_probs=self.end_probs._lookup)

    def sample(self, n_samples):
        encoding = self.transition_matrix.encoding
        n_letters = encoding.alphabet_size
        transitions = np.exp(self.transition_matrix.raw())
        initial_probs = np.exp(self.initial_distribution.raw())
        end_probs = np.exp(self.end_probs.raw())
        matrix = np.hstack((transitions, end_probs[:, None]))
        matrix = np.concatenate([matrix, [np.append(initial_probs, 0)]])
        state = -1
        choices = np.arange(n_letters+1)
        data = []
        lens = []
        len_counter = 0
        while len(lens) < n_samples:
            assert np.allclose(matrix[state].sum(), 1), matrix[state]
            state = np.random.choice(choices, p=matrix[state])
            if state == n_letters:
                lens.append(len_counter)
                len_counter = 0
                continue
            len_counter += 1
            data.append(state)
        assert len(lens) == n_samples, (len(lens), n_samples)
        return EncodedRaggedArray(
            EncodedArray(np.array(data, dtype=int), encoding),
            lens)

    @classmethod
    def load(cls, filename):
        encoding = AminoAcidEncoding
        data = np.load(filename)
        return cls(EncodedLookup(data['transition_matrix'], encoding),
                   EncodedLookup(data['initial_distribution'], encoding),
                   EncodedLookup(data['end_probs'], encoding))


def estimate_transition_model(sequences, weights=None):
    pseudo_count = 1
    n = len(sequences)
    encoding: AlphabetEncoding = sequences.encoding
    n_letters = encoding.alphabet_size
    transition_pseudo = pseudo_count*np.mean(sequences.lengths)/(n_letters + 1)/n_letters
    initial_distribution = np.bincount(sequences[..., 0].raw(), weights=weights, minlength=n_letters) + pseudo_count/n_letters
    end_probs = np.bincount(sequences[..., -1].raw(), weights=weights, minlength=n_letters) + transition_pseudo
    # denominator = np.bincount(sequences.raw().ravel(), minlength=n_letters) + pseudo_count
    if weights is not None:
        weights = RaggedShape(sequences.lengths - 1).broadcast_values(weights[:, None]).ravel()
    transition_matrix = np.bincount(
        np.ravel_multi_index((sequences[..., :-1].raw().ravel(), sequences[..., 1:].raw().ravel()),
                             (n_letters, n_letters)),
        weights=weights, minlength=n_letters ** 2).reshape((n_letters, n_letters)) + transition_pseudo
    row_sums = transition_matrix.sum(axis=1)+end_probs
    transition_matrix = transition_matrix / row_sums[..., None]
    initial_distribution = initial_distribution / initial_distribution.sum()
    end_probs = end_probs / row_sums
    prob_sum = np.sum(transition_matrix, axis=-1) + end_probs
    assert np.allclose(prob_sum, 1), (np.sum(transition_matrix, axis=-1), end_probs)

    return SequenceTransitionDistribution.from_probabilities(
        EncodedLookup(transition_matrix, encoding),
        EncodedLookup(initial_distribution, encoding),
        EncodedLookup(end_probs, encoding))
