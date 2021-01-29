import copy
from typing import List

import pandas as pd

from immuneML.IO.dataset_import.PickleImport import PickleImport
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.SimulationState import SimulationState
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.Step import Step


class SignalImplanter(Step):

    DATASET_NAME = "simulated_dataset"

    @staticmethod
    def run(simulation_state: SimulationState = None):
        path = simulation_state.result_path / FilenameHandler.get_dataset_name(SignalImplanter.__name__)

        if path.is_file():
            dataset = PickleImport.import_dataset({"path": path}, SignalImplanter.DATASET_NAME)
        else:
            dataset = SignalImplanter._implant_signals_in_dataset(simulation_state)

        return dataset

    @staticmethod
    def _implant_signals_in_dataset(simulation_state: SimulationState = None) -> Dataset:
        PathBuilder.build(simulation_state.result_path)

        if isinstance(simulation_state.dataset, RepertoireDataset):
            dataset = SignalImplanter._implant_signals_in_repertoires(simulation_state)
        else:
            dataset = SignalImplanter._implant_signals_in_receptors(simulation_state)

        return dataset

    @staticmethod
    def _implant_signals_in_receptors(simulation_state: SimulationState) -> Dataset:
        processed_receptors = SignalImplanter._implant_signals(simulation_state, SignalImplanter._process_receptor)
        processed_dataset = ReceptorDataset.build(receptors=processed_receptors, file_size=simulation_state.dataset.file_size,
                                                  name=simulation_state.dataset.name, path=simulation_state.result_path)

        processed_dataset.labels = {**(simulation_state.dataset.labels if simulation_state.dataset.labels is not None else {}),
                                    **{signal: [True, False] for signal in simulation_state.signals}}

        return processed_dataset

    @staticmethod
    def _implant_signals_in_repertoires(simulation_state: SimulationState = None) -> Dataset:

        PathBuilder.build(simulation_state.result_path / "repertoires")
        processed_repertoires = SignalImplanter._implant_signals(simulation_state, SignalImplanter._process_repertoire)
        processed_dataset = RepertoireDataset(repertoires=processed_repertoires, labels={**(simulation_state.dataset.labels if simulation_state.dataset.labels is not None else {}),
                                                                                         **{signal.id: [True, False] for signal in simulation_state.signals}},
                                              name=simulation_state.dataset.name,
                                              metadata_file=SignalImplanter._create_metadata_file(processed_repertoires, simulation_state))
        return processed_dataset

    @staticmethod
    def _implant_signals(simulation_state: SimulationState, process_element_func):
        processed_elements = []
        simulation_limits = SignalImplanter._prepare_simulation_limits(simulation_state.simulation.implantings,
                                                                       simulation_state.dataset.get_example_count())
        current_implanting_index = 0
        current_implanting = simulation_state.simulation.implantings[current_implanting_index]

        for index, element in enumerate(simulation_state.dataset.get_data()):

            if current_implanting is not None and index >= simulation_limits[current_implanting.name]:
                current_implanting_index += 1
                if current_implanting_index < len(simulation_limits.keys()):
                    current_implanting = simulation_state.simulation.implantings[current_implanting_index]
                else:
                    current_implanting = None

            processed_element = process_element_func(index, element, current_implanting, simulation_state)
            processed_elements.append(processed_element)

        return processed_elements

    @staticmethod
    def _process_receptor(index, receptor, implanting, simulation_state) -> Receptor:
        if implanting is not None:
            new_receptor = simulation_state.signals[0].implant_in_receptor(receptor, implanting.is_noise)
        else:
            new_receptor = receptor.clone()
            for signal in simulation_state.signals:
                new_receptor.metadata[f"signal_{signal.id}"] = False
        return new_receptor

    @staticmethod
    def _process_repertoire(index, repertoire, current_implanting, simulation_state) -> Repertoire:
        if current_implanting is not None:

            return SignalImplanter._implant_in_repertoire(index, repertoire, current_implanting, simulation_state)

        else:
            new_repertoire = Repertoire.build_from_sequence_objects(repertoire.sequences, simulation_state.result_path / "repertoires",
                                                                    repertoire.metadata)

            for signal in simulation_state.signals:
                new_repertoire.metadata[f"signal_{signal.id}"] = False

            return new_repertoire

    @staticmethod
    def _create_metadata_file(processed_repertoires: List[Repertoire], simulation_state) -> str:

        path = simulation_state.result_path / "metadata.csv"

        new_df = pd.DataFrame([repertoire.metadata for repertoire in processed_repertoires])
        new_df.drop('field_list', axis=1, inplace=True)
        new_df["filename"] = [repertoire.data_filename for repertoire in processed_repertoires]
        new_df.to_csv(path, index=False)

        return path

    @staticmethod
    def _implant_in_repertoire(index, repertoire, implanting, simulation_state) -> Repertoire:
        new_repertoire = copy.deepcopy(repertoire)
        for signal in implanting.signals:
            new_repertoire = signal.implant_to_repertoire(repertoire=new_repertoire,
                                                          repertoire_implanting_rate=implanting.repertoire_implanting_rate,
                                                          path=simulation_state.result_path / "repertoires/")

        for signal in implanting.signals:
            if implanting.is_noise:
                new_repertoire.metadata[f"signal_{signal.id}"] = False
            else:
                new_repertoire.metadata[f"signal_{signal.id}"] = True
        for signal in simulation_state.signals:
            if signal not in implanting.signals:
                new_repertoire.metadata[f"signal_{signal.id}"] = False

        return new_repertoire

    @staticmethod
    def _prepare_simulation_limits(simulation: list, element_count: int) -> dict:
        """for each implanting returns the last index of the element in the dataset with that implanting scheme"""
        limits = {item.name: int(item.dataset_implanting_rate * element_count) for item in simulation}
        limits = {item_name: sum(list(limits.values())[:i+1]) for i, item_name in enumerate(limits.keys())}

        return limits
