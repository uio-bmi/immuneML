import copy
import os

import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.SignalImplanterParams import SignalImplanterParams
from source.workflows.steps.Step import Step


class SignalImplanter(Step):

    @staticmethod
    def run(input_params: SignalImplanterParams = None):
        return SignalImplanter.perform_step(input_params)

    @staticmethod
    def perform_step(input_params: SignalImplanterParams = None):

        path = input_params.result_path + FilenameHandler.get_dataset_name(SignalImplanter.__name__)

        if os.path.isfile(path):
            dataset = PickleLoader.load(path)
        else:
            dataset = SignalImplanter._implant_signals(input_params)

        return dataset

    @staticmethod
    def _implant_signals(input_params: SignalImplanterParams = None) -> Dataset:

        PathBuilder.build(input_params.result_path)

        processed_repertoires = []
        simulation_limits = SignalImplanter._prepare_simulation_limits(input_params.simulation.implantings,
                                                                       input_params.dataset.get_example_count())
        simulation_index = 0

        implanting_metadata = {f"signal_{signal.id}": [] for signal in input_params.signals}

        for index, repertoire in enumerate(input_params.dataset.get_data(input_params.batch_size)):

            if simulation_index <= len(simulation_limits) - 1 and index >= simulation_limits[simulation_index]:
                simulation_index += 1

            processed_repertoire = SignalImplanter._process_repertoire(index, repertoire, simulation_index, simulation_limits, input_params)
            processed_repertoires.append(processed_repertoire)

            for signal in input_params.signals:
                implanting_metadata[f"signal_{signal.id}"].append(processed_repertoire.metadata[f"signal_{signal.id}"])

        processed_dataset = RepertoireDataset(repertoires=processed_repertoires, params=input_params.dataset.params,
                                              metadata_file=SignalImplanter._create_metadata_file(input_params.dataset.metadata_file,
                                                                                                  implanting_metadata, input_params))
        PickleExporter.export(processed_dataset, input_params.result_path, FilenameHandler.get_dataset_name(SignalImplanter.__name__))

        return processed_dataset

    @staticmethod
    def _create_metadata_file(metadata_path, implanting_metadata: dict, input_params) -> str:

        new_info_df = pd.DataFrame(implanting_metadata)
        path = input_params.result_path + "metadata.csv"

        if metadata_path:
            df = pd.read_csv(metadata_path)
        else:
            df = pd.DataFrame({"filename": input_params.dataset.get_example_ids()})

        new_df = pd.concat([df, new_info_df], axis=1)
        new_df.to_csv(path, index=False)

        return path

    @staticmethod
    def _process_repertoire(index, repertoire, simulation_index, simulation_limits, input_params):
        if simulation_index < len(simulation_limits):
            return SignalImplanter._implant_in_repertoire(index, repertoire, simulation_index, input_params)
        else:
            return SignalImplanter._copy_repertoire(index, repertoire, input_params)

    @staticmethod
    def _copy_repertoire(index: int, repertoire: Repertoire, input_params: SignalImplanterParams) -> str:
        new_repertoire = Repertoire.build_from_sequence_objects(repertoire.sequences, input_params.result_path, repertoire.metadata)

        for signal in input_params.signals:
            new_repertoire.metadata[f"signal_{signal.id}"] = False

        return new_repertoire

    @staticmethod
    def _implant_in_repertoire(index, repertoire, simulation_index, input_params) -> str:
        new_repertoire = copy.deepcopy(repertoire)
        for signal in input_params.simulation.implantings[simulation_index].signals:
            new_repertoire = signal.implant_to_repertoire(repertoire=new_repertoire,
                                                          repertoire_implanting_rate=
                                                          input_params.simulation.implantings[simulation_index].repertoire_implanting_rate,
                                                          path=input_params.result_path)

        for signal in input_params.simulation.implantings[simulation_index].signals:
            new_repertoire.metadata[f"signal_{signal.id}"] = True
        for signal in input_params.signals:
            if signal not in input_params.simulation.implantings[simulation_index].signals:
                new_repertoire.metadata[f"signal_{signal.id}"] = False

        return new_repertoire

    @staticmethod
    def _prepare_simulation_limits(simulation: list, repertoire_count: int) -> list:
        limits = [int(item.dataset_implanting_rate * repertoire_count) for item in simulation]
        limits = [sum(limits[:i+1]) for i in range(len(limits))]
        return limits
