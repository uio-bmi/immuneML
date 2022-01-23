from pathlib import Path
from typing import List

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationState import SimulationState
from immuneML.util.ExporterHelper import ExporterHelper
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

        store_signal_in_receptors (bool): for repertoire-level simulation, whether to store the information on what exact motif is implanted in each receptor

        export_formats: in which formats to export the dataset after simulation. Valid formats are class names of any non-abstract class inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`. Important note: Binary files in ImmuneML might not be compatible between different immuneML versions.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simulation_instruction: # user-defined name of the instruction
            type: Simulation # which instruction to execute
            dataset: my_dataset # which dataset to use for implanting the signals
            simulation: my_simulation # how to implanting the signals - definition of the simulation
            export_formats: [AIRR] # in which formats to export the dataset
            store_signal_in_receptors: True

    """

    def __init__(self, signals: list, simulation: Simulation, dataset: RepertoireDataset, store_signal_in_receptors: bool,
                 name: str = None, exporters: List[DataExporter] = None):
        self.exporters = exporters
        self.state = SimulationState(signals=signals, simulation=simulation, dataset=dataset, name=name,
                                     store_signal_in_receptors=store_signal_in_receptors)

    def run(self, result_path: Path):
        self.state.result_path = result_path / self.state.name
        self.state.resulting_dataset = SignalImplanter.run(self.state)
        export_output = ExporterHelper.export_dataset(self.state.resulting_dataset, self.exporters, self.state.result_path)
        self.state.formats = export_output['formats']
        self.state.paths = export_output['paths']
        return self.state

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
