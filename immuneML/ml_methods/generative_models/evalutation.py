def evaluate_similarities(true_sequences, simulated_sequences, estimator):
    true_model = estimator(true_sequences)
    simulated_model = estimator(simulated_sequences)

