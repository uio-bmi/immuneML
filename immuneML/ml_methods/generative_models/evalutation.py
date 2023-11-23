from functools import lru_cache

import numpy as np
import pandas as pd


def KL(sequences, model_1, model_2):
    """
    Computes the KL divergence between two models (model_1 and model_2) for a given set of sequences.

    Args:
        sequences: list of sequences
        model_1: model 1
        model_2: model 2

    Returns:
        KL divergence value
    """
    return np.mean(get_kl_weights(model_1, model_2, sequences))


def get_kl_weights(model_1, model_2, sequences):
    return model_1.log_prob(sequences) - model_2.log_prob(sequences)


def evaluate_similarities(true_sequences, simulated_sequences, estimator):
    true_model = estimator(true_sequences)
    simulated_model = estimator(simulated_sequences)
    return KL(true_sequences, true_model, simulated_model), \
           KL(simulated_sequences, simulated_model, true_model)


class KLEvaluator:
    def __init__(self, true_sequences, simulated_sequences, estimator):
        self.true_sequences = true_sequences
        self.simulated_sequences = simulated_sequences
        self.true_model = estimator(true_sequences)
        self.simulated_model = estimator(simulated_sequences)
        self.estimator = estimator

    @lru_cache()
    def true_kl_weights(self):
        return get_kl_weights(self.true_model, self.simulated_model, self.true_sequences)

    @lru_cache()
    def simulated_kl_weights(self):
        return get_kl_weights(self.simulated_model, self.true_model, self.simulated_sequences)

    def true_kl(self):
        return np.mean(self.true_kl_weights())

    def simulated_kl(self):
        return np.mean(self.simulated_kl_weights())

    def get_worst_true_sequences(self, n=20):
        indices = np.argsort(self.true_kl_weights())[-n:][::-1]
        return pd.DataFrame({"sequence": self.true_sequences[indices].tolist(),
                             "kl": self.true_kl_weights()[indices]})
        # return self.true_sequences[indices]

    def get_worst_simulated_sequences(self, n=20):
        indices = np.argsort(self.simulated_kl_weights())[-n:][::-1]
        return pd.DataFrame({"sequence": self.simulated_sequences[indices].tolist(),
                                "kl": self.simulated_kl_weights()[indices]})
