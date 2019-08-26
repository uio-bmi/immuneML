from source.data_model.dataset.Dataset import Dataset


class SignalImplanterParams:

    def __init__(self, dataset: Dataset, result_path: str, simulations: list, signals: list, batch_size: int = 1):
        self.dataset = dataset
        self.result_path = result_path
        self.simulations = simulations
        self.signals = signals
        self.batch_size = batch_size
