from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.simulation.Simulation import Simulation
from source.simulation.SimulationState import SimulationState
from source.workflows.instructions.Instruction import Instruction
from source.workflows.steps.SignalImplanter import SignalImplanter


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

    def __init__(self, signals: list, simulation: Simulation, dataset: RepertoireDataset, path: str = None, batch_size: int = 1,
                 name: str = None):
        self.state = SimulationState(signals, simulation, dataset, path=path, batch_size=batch_size, name=name)

    def run(self, result_path: str):
        self.state.result_path = result_path
        self.state.resulting_dataset = SignalImplanter.run(self.state)
        return self.state
