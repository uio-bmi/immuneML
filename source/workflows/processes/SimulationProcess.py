from source.data_model.dataset.Dataset import Dataset
from source.workflows.processes.InstructionProcess import InstructionProcess
from source.workflows.steps.SignalImplanter import SignalImplanter
from source.workflows.steps.SignalImplanterParams import SignalImplanterParams


class SimulationProcess(InstructionProcess):
    """
    Simulation process implants signals into the given dataset according to given parameters;
    it supports multiple simulation definitions and relies on SignalImplanter to perform the work
    """

    def __init__(self, signals: list, simulations: list, dataset: Dataset, path: str = None, batch_size: int = 1):
        self.signals = signals
        self.simulations = simulations
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size

    def run(self):
        dataset = SignalImplanter.run(SignalImplanterParams(dataset=self.dataset, result_path=self.path,
                                                            simulations=self.simulations, signals=self.signals,
                                                            batch_size=self.batch_size))

        return {
            "repertoires": dataset.get_example_count(),
            "result_path": dataset.metadata_file,
            "signals": [str(s) for s in self.signals],
            "simulations": [str(s) for s in self.simulations]
        }
