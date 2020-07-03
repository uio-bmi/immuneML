from typing import List

from scripts.specification_util import update_docs_per_mapping
from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.simulation.Simulation import Simulation
from source.simulation.SimulationState import SimulationState
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.Instruction import Instruction
from source.workflows.steps.SignalImplanter import SignalImplanter


class SimulationInstruction(Instruction):
    """
    A Simulation is an instruction that implants synthetic signals into the given dataset according
    to given parameters. This results in a new dataset containing modified sequences, and is annotated
    with metadata labels according to the implanted signals.

    Arguments:

        dataset (RepertoireDataset): original dataset which will be used as a basis for implanting signals from the simulation

        simulation (Simulation): definition of how to perform the simulation.

        batch_size (int): how many parallel processes to use during the analysis (4 is usually a good choice for personal computers).

        exporter: in which format to export the dataset after simulation. Valid formats are class names of any non-abstract class inheriting :py:obj:`~source.IO.dataset_export.DataExporter.DataExporter`.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simulation_instruction: # user-defined name of the instruction
            type: Simulation # which instruction to execute
            dataset: my_dataset # which dataset to use for implanting the signals
            simulation: my_simulation # how to implanting the signals - definition of the simulation
            batch_size: 4 # how many parallel processes to use during execution
            export_formats: [AIRR] # in which formats to export the dataset

    """

    def __init__(self, signals: list, simulation: Simulation, dataset: RepertoireDataset, path: str = None, batch_size: int = 1,
                 name: str = None, exporters: List[DataExporter] = None):
        self.state = SimulationState(signals, simulation, dataset, path=path, batch_size=batch_size, name=name)
        self.exporters = exporters

    def run(self, result_path: str):
        self.state.result_path = result_path
        self.state.resulting_dataset = SignalImplanter.run(self.state)
        self.export_dataset()
        return self.state

    def export_dataset(self):
        if self.exporters is not None and len(self.exporters) > 0:
            for exporter in self.exporters:
                exporter.export(self.state.resulting_dataset,
                                f"{self.state.result_path}exported_dataset/{exporter.__name__.replace('Exporter', '').lower()}/")

    @staticmethod
    def get_documentation():
        doc = str(SimulationInstruction.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, "Exporter", "dataset_export/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        mapping = {
            "Valid formats are class names of any non-abstract class inheriting "
            ":py:obj:`~source.IO.dataset_export.DataExporter.DataExporter`.": f"Valid values are: {valid_strategy_values}.",
            "exporter": "export_format",
            "simulation (Simulation)": "simulation",
            "dataset (RepertoireDataset)": "dataset"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
