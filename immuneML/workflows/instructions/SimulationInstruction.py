from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationState import SimulationState
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.steps.SignalImplanter import SignalImplanter
from scripts.specification_util import update_docs_per_mapping


class SimulationInstruction(Instruction):
    """
    A simulation is an instruction that implants synthetic signals into the given dataset according
    to given parameters. This results in a new dataset containing modified sequences, and is annotated
    with metadata labels according to the implanted signals.

    Arguments:

        dataset (RepertoireDataset): original dataset which will be used as a basis for implanting signals from the simulation

        simulation (Simulation): definition of how to perform the simulation.

        export_formats: in which formats to export the dataset after simulation. Valid formats are class names of any non-abstract class inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simulation_instruction: # user-defined name of the instruction
            type: Simulation # which instruction to execute
            dataset: my_dataset # which dataset to use for implanting the signals
            simulation: my_simulation # how to implanting the signals - definition of the simulation
            export_formats: [AIRR] # in which formats to export the dataset

    """

    def __init__(self, signals: list, simulation: Simulation, dataset: RepertoireDataset,
                 name: str = None, exporters: List[DataExporter] = None):
        self.exporters = exporters
        self.state = SimulationState(signals, simulation, dataset, name=name)

    def run(self, result_path: Path):
        self.state.result_path = result_path / self.state.name
        self.state.resulting_dataset = SignalImplanter.run(self.state)
        self.export_dataset()
        return self.state

    def export_dataset(self):
        dataset_name = self.state.resulting_dataset.name if self.state.resulting_dataset.name is not None else self.state.resulting_dataset.identifier
        paths = {dataset_name: {}}
        formats = []

        if self.exporters is not None and len(self.exporters) > 0:
            for exporter in self.exporters:
                export_format = exporter.__name__[:-8]
                path = self.state.result_path / f"exported_dataset/{exporter.__name__.replace('Exporter', '').lower()}/"
                exporter.export(self.state.resulting_dataset,
                                path)
                paths[dataset_name][export_format] = path
                formats.append(export_format)

        self.state.paths = paths
        self.state.formats = formats

    @staticmethod
    def get_documentation():
        doc = str(SimulationInstruction.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, "Exporter", "dataset_export/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        mapping = {
            "Valid formats are class names of any non-abstract class inheriting "
            ":py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`.": f"Valid values are: {valid_strategy_values}.",
            "exporter": "export_format",
            "simulation (Simulation)": "simulation",
            "dataset (RepertoireDataset)": "dataset"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
