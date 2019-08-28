import copy
import os
import pickle

import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
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

        processed_filenames = []
        simulation_limits = SignalImplanter._prepare_simulation_limits(input_params.simulations,
                                                                       input_params.dataset.get_example_count())
        simulation_index = 0

        implanting_metadata = {signal.id: [] for signal in input_params.signals}

        for index, repertoire in enumerate(input_params.dataset.get_data(input_params.batch_size)):

            if simulation_index <= len(simulation_limits) - 1 and index >= simulation_limits[simulation_index]:
                simulation_index += 1

            filename = SignalImplanter._process_repertoire(index, repertoire, simulation_index, simulation_limits, input_params)
            processed_filenames.append(filename)

            rep = input_params.dataset.get_repertoire(filename=filename)
            for signal in input_params.signals:
                implanting_metadata[signal.id].append(rep.metadata.custom_params[signal.id])

        processed_dataset = RepertoireDataset(filenames=processed_filenames, params=input_params.dataset.params,
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
            df = pd.DataFrame({"filename": input_params.dataset.get_filenames()})

        new_df = pd.concat([df, new_info_df], axis=1)
        new_df.to_csv(path, index=False)

        return path

    @staticmethod
    def _process_repertoire(index, repertoire, simulation_index, simulation_limits, input_params):

        if simulation_index < len(simulation_limits):
            filename = SignalImplanter._implant_in_repertoire(index, repertoire, simulation_index, input_params)
        else:
            filename = SignalImplanter._copy_repertoire(index, repertoire, input_params)

        return filename

    @staticmethod
    def _copy_repertoire(index: int, repertoire: SequenceRepertoire, input_params: SignalImplanterParams) -> str:
        new_repertoire = copy.deepcopy(repertoire)
        if new_repertoire.metadata is None:
            new_repertoire.metadata = RepertoireMetadata(sample=Sample(identifier=""))

        for signal in input_params.signals:
            new_repertoire.metadata.custom_params[signal.id] = False

        filename = input_params.result_path + "rep" + str(index) + ".pickle"

        with open(filename, "wb") as file:
            pickle.dump(new_repertoire, file)

        return filename

    @staticmethod
    def _implant_in_repertoire(index, repertoire, simulation_index, input_params) -> str:
        new_repertoire = repertoire
        for signal in input_params.simulations[simulation_index].signals:
            new_repertoire = signal.implant_to_repertoire(repertoire=new_repertoire,
                                                          repertoire_implanting_rate=
                                                          input_params.simulations[simulation_index].repertoire_implanting_rate)

        for signal in input_params.simulations[simulation_index].signals:
            new_repertoire.metadata.custom_params[signal.id] = True
        for signal in input_params.signals:
            if signal not in input_params.simulations[simulation_index].signals:
                new_repertoire.metadata.custom_params[signal.id] = False

        filename = input_params.result_path + "rep" + str(index) + ".pickle"

        with open(filename, "wb") as file:
            pickle.dump(new_repertoire, file)

        return filename

    @staticmethod
    def _prepare_simulation_limits(simulation: list, repertoire_count: int) -> list:
        limits = [int(item.dataset_implanting_rate * repertoire_count) for item in simulation]
        limits = [sum(limits[:i+1]) for i in range(len(limits))]
        return limits
