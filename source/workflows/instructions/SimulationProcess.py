from source.data_model.dataset.Dataset import Dataset
from source.simulation.Simulation import Simulation
from source.workflows.instructions.Instruction import Instruction
from source.workflows.steps.SignalImplanter import SignalImplanter
from source.workflows.steps.SignalImplanterParams import SignalImplanterParams


class SimulationInstruction(Instruction):
    """
    A Simulation is an instruction that implants synthetic signals into the given dataset according
    to given parameters. This results in a new dataset containing modified sequences, and is annotated
    with metadata labels according to the implanted signals.


    Specification:

        definitions:
            datasets:
                my_dataset:
                    ...

            motifs:
                my_motif:
                    ...

            signals:
                my_signal:
                    motifs:
                        - my_motif
                        - ...
                    implanting: HealthySequence
                    ...

            simulation:
                my_simulation:
                    my_implanting:
                        signals:
                            - my_signal
                        ...
        instructions:
            my_simulation_instruction:
                type: Simulation
                dataset: my_dataset
                simulation: my_simulation
                batch_size: 5
    """

    def __init__(self, signals: list, simulation: Simulation, dataset: Dataset, path: str = None, batch_size: int = 1):
        self.signals = signals
        self.simulation = simulation
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size

    def run(self, result_path: str):
        self.path = result_path
        dataset = SignalImplanter.run(SignalImplanterParams(dataset=self.dataset, result_path=self.path,
                                                            simulation=self.simulation, signals=self.signals,
                                                            batch_size=self.batch_size))

        return {
            "repertoires": dataset.get_example_count(),
            "result_path": dataset.metadata_file,
            "signals": [str(s) for s in self.signals],
            "simulations": [str(s) for s in self.simulation.implantings]
        }
