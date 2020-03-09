from source.data_model.dataset.Dataset import Dataset
from source.simulation.Simulation import Simulation


class SignalImplanterParams:

    def __init__(self, dataset: Dataset, result_path: str, simulation: Simulation, signals: list, batch_size: int = 1):
        self.dataset = dataset
        self.result_path = result_path
        self.simulation = simulation
        self.signals = signals
        self.batch_size = batch_size
