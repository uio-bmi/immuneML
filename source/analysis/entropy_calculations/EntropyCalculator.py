import numpy as np


class EntropyCalculator:
    """
    Set of functions to help with diversity calculations
    """
    @staticmethod
    def shannon_entropy(x):
        a = np.array(x) if not isinstance(x, np.ndarray) else x
        a = a[np.nonzero(a)]
        p = a / np.sum(a)
        return -np.sum(p * np.log(p))

    @staticmethod
    def min_entropy(x):
        a = np.array(x) if not isinstance(x, np.ndarray) else x
        a = a[np.nonzero(a)]
        f = a / np.sum(a)
        return -np.log(np.max(f))

    @staticmethod
    def renyi_entropy(x, alpha):
        a = np.array(x) if not isinstance(x, np.ndarray) else x
        a = a[np.nonzero(a)]
        f = a / np.sum(a)
        if np.abs(alpha - 1) < 1e-10:
            entropy = EntropyCalculator.shannon_entropy(a)
        elif alpha == np.inf:
            entropy = EntropyCalculator.min_entropy(a)
        else:
            entropy = (1 / (1 - alpha)) * np.log(np.sum(f ** alpha))
        return entropy
