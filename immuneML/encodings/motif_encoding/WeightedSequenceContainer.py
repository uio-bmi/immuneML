


class WeigthedSequenceContainer:
    def __init__(self, np_sequences, weights, y_true):
        self._check_lengths(np_sequences, weights, y_true)

        self.np_sequences = np_sequences
        self.weights = weights
        self.y_true = y_true

    def _check_lengths(self, np_sequences, weights, y_true):
        assert len(np_sequences) == len(y_true), \
            f"NumpySequenceContainer: np_sequences and y_true must have the same length. " \
            f"Found: {len(np_sequences)} {len(y_true)}"

        if weights is not None:
            assert len(np_sequences) == len(y_true), \
                f"NumpySequenceContainer: np_sequences and weights must have the same length. " \
                f"Found: {len(np_sequences)} {len(weights)}"

    def __len__(self):
        return len(self.np_sequences)