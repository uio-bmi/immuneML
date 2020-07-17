import copy
import os
from typing import List

import pandas as pd

from source.IO.dataset_import.PickleImport import PickleImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.Receptor import Receptor
from source.data_model.repertoire.Repertoire import Repertoire
from source.simulation.SimulationState import SimulationState
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.Step import Step


class SignalImplanter(Step):

    DATASET_NAME = "simulated_dataset"

    @staticmethod
    def run(simulation_state: SimulationState = None):
        path = simulation_state.result_path + FilenameHandler.get_dataset_name(SignalImplanter.__name__)

        if os.path.isfile(path):
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
        return processed_dataset

    @staticmethod
    def _implant_signals_in_repertoires(simulation_state: SimulationState = None) -> Dataset:

        PathBuilder.build(simulation_state.result_path + "repertoires/")
        processed_repertoires = SignalImplanter._implant_signals(simulation_state, SignalImplanter._process_repertoire)
        processed_dataset = RepertoireDataset(repertoires=processed_repertoires, params=simulation_state.dataset.params, name=simulation_state.dataset.name,
                                              metadata_file=SignalImplanter._create_metadata_file(processed_repertoires, simulation_state))
        return processed_dataset

    @staticmethod
    def _implant_signals(simulation_state: SimulationState, process_element_func):
        processed_elements = []
        simulation_limits = SignalImplanter._prepare_simulation_limits(simulation_state.simulation.implantings,
                                                                       simulation_state.dataset.get_example_count())
        simulation_index = 0

        for index, element in enumerate(simulation_state.dataset.get_data(simulation_state.batch_size)):

            if simulation_index <= len(simulation_limits) - 1 and index >= simulation_limits[simulation_index]:
                simulation_index += 1

            processed_element = process_element_func(index, element, simulation_index, simulation_limits, simulation_state)
            processed_elements.append(processed_element)

        return processed_elements

    @staticmethod
    def _process_receptor(index, receptor, simulation_index, simulation_limits, simulation_state) -> Receptor:
        if simulation_index < len(simulation_limits):
            return simulation_state.signals[0].implant_in_receptor(receptor)
        else:
            new_receptor = receptor.clone()
            new_receptor.metadata[f"signal_{simulation_state.signals[0].id}"] = False
            return new_receptor

    @staticmethod
    def _process_repertoire(index, repertoire, simulation_index, simulation_limits, simulation_state) -> Repertoire:
        if simulation_index < len(simulation_limits):
            return SignalImplanter._implant_in_repertoire(index, repertoire, simulation_index, simulation_state)
        else:
            return SignalImplanter._copy_repertoire(index, repertoire, simulation_state)

    @staticmethod
    def _create_metadata_file(processed_repertoires: List[Repertoire], simulation_state) -> str:

        path = simulation_state.result_path + "metadata.csv"

        new_df = pd.DataFrame([repertoire.metadata for repertoire in processed_repertoires])
        new_df.drop('field_list', axis=1, inplace=True)
        new_df["filename"] = [repertoire.data_filename for repertoire in processed_repertoires]
        new_df.to_csv(path, index=False)

        return path

    @staticmethod
    def _copy_repertoire(index: int, repertoire: Repertoire, simulation_state: SimulationState) -> Repertoire:
        new_repertoire = Repertoire.build_from_sequence_objects(repertoire.sequences, simulation_state.result_path + "repertoires/", repertoire.metadata)

        for signal in simulation_state.signals:
            new_repertoire.metadata[f"signal_{signal.id}"] = False

        return new_repertoire

    @staticmethod
    def _implant_in_repertoire(index, repertoire, simulation_index, simulation_state) -> Repertoire:
        new_repertoire = copy.deepcopy(repertoire)
        for signal in simulation_state.simulation.implantings[simulation_index].signals:
            new_repertoire = signal.implant_to_repertoire(repertoire=new_repertoire,
                                                          repertoire_implanting_rate=
                                                          simulation_state.simulation.implantings[simulation_index].repertoire_implanting_rate,
                                                          path=simulation_state.result_path + "repertoires/")

        for signal in simulation_state.simulation.implantings[simulation_index].signals:
            new_repertoire.metadata[f"signal_{signal.id}"] = True
        for signal in simulation_state.signals:
            if signal not in simulation_state.simulation.implantings[simulation_index].signals:
                new_repertoire.metadata[f"signal_{signal.id}"] = False

        return new_repertoire

    @staticmethod
    def _prepare_simulation_limits(simulation: list, element_count: int) -> list:
        limits = [int(item.dataset_implanting_rate * element_count) for item in simulation]
        limits = [sum(limits[:i+1]) for i in range(len(limits))]
        return limits
