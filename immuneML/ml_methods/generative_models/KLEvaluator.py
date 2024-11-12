import plotly.graph_objects as go
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
    def __init__(self, true_sequences, simulated_sequences, estimator, n_sequences):
        self.true_sequences = true_sequences
        self.simulated_sequences = simulated_sequences
        self.true_model = estimator(true_sequences)
        self.simulated_model = estimator(simulated_sequences)
        self.estimator = estimator
        self._n_sequences = n_sequences

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
        indices = self._simulated_indices(n)
        return pd.DataFrame({"sequence": self.simulated_sequences[indices].tolist(),
                                "kl": self.simulated_kl_weights()[indices]})

    def _simulated_indices(self, n):
        return np.argsort(self.simulated_kl_weights())[-n:][::-1]

    def simulated_plot(self):
        n_sequences = self._n_sequences
        indices = self._simulated_indices(n_sequences)
        kmers = self.simulated_sequences[indices]
        weights = self.simulated_model.kmer_model.log_prob(kmers) - self.true_model.kmer_model.log_prob(kmers)
        scores = self.simulated_kl_weights()[indices]
        fig = self.get_plot(indices, kmers, scores, weights)
        fig.update_layout(yaxis_title="Sequence", xaxis_title="Position",
                          title="KL weights for the generated sequences that don't fit with the original model")
        return fig

    def original_plot(self):
        n_sequences = self._n_sequences
        indices = np.argsort(self.true_kl_weights())[-n_sequences:][::-1]
        kmers = self.true_sequences[indices]
        weights = self.true_model.kmer_model.log_prob(kmers) - self.simulated_model.kmer_model.log_prob(kmers)
        scores = self.true_kl_weights()[indices]
        fig = self.get_plot(indices, kmers, scores, weights)
        fig.update_layout(yaxis_title="Sequence", xaxis_title="Position",
                          title="KL weights for the original sequences that don't fit the generated model")
        # add label for colorbar
        #fig.update_layout(colorbar={"title": 'Your title'})

        return fig


    def get_plot(self, indices, kmers, scores, weights):
        n_sequences = len(indices)
        width = np.max(weights.shape[-1])
        z = np.full((n_sequences, width), np.nan)
        text = [['' for _ in range(width)] for _ in range(n_sequences)]
        sequences = []
        for kmers, weight, row, text_row, score in zip(kmers, weights, z, text, scores):
            row[:len(kmers)] = weight
            text_row[:len(kmers)] = [kmer.to_string() for kmer in kmers]

            seq = ''.join([kmers[0].to_string()[:-1]] + [kmer.to_string()[-1] for kmer in kmers])
            sequences.append(f'{seq} : {score:.2f}')
        fig = go.Figure(data=go.Heatmap(
            y=sequences,
            z=z,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="KL weights"),
            colorscale=["green", 'yellow', 'red']
        ))
        fig.update_yaxes(autorange="reversed")
        return fig