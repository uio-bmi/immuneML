from source.IO.dataset_export.DataExporter import DataExporter
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
                export_format: AIRR
    """

    def __init__(self, signals: list, simulation: Simulation, dataset: RepertoireDataset, path: str = None, batch_size: int = 1,
                 name: str = None, exporter: DataExporter = None):
        self.state = SimulationState(signals, simulation, dataset, path=path, batch_size=batch_size, name=name)
        self.exporter = exporter

    def run(self, result_path: str):
        self.state.result_path = result_path
        self.state.resulting_dataset = SignalImplanter.run(self.state)
        self.export_dataset()
        return self.state

    def export_dataset(self):
        if self.exporter is not None:
            self.exporter.export(self.state.resulting_dataset, f"{self.state.result_path}exported_dataset/")
