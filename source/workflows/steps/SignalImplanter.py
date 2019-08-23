import copy
import os
import pickle

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.util.FilenameHandler import FilenameHandler
from source.util.PathBuilder import PathBuilder
from source.workflows.steps.Step import Step


class SignalImplanter(Step):

    @staticmethod
    def run(input_params: dict = None):
        SignalImplanter.check_prerequisites(input_params)
        return SignalImplanter.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert "simulation" in input_params, "SignalImplanterStep: specify the simulation parameter."
        assert "dataset" in input_params and isinstance(input_params["dataset"], RepertoireDataset), "SignalImplanterStep: specify the dataset parameter."
        assert "result_path" in input_params, "SignalImplanterStep: specify the result_path parameter."
        assert "batch_size" in input_params, "SignalImplanterStep: specify the batch_size parameter for loading repertoires."

    @staticmethod
    def perform_step(input_params: dict = None):

        path = input_params["result_path"] + FilenameHandler.get_dataset_name(SignalImplanter.__name__)

        if os.path.isfile(path):
            dataset = PickleLoader.load(path)
        else:
            dataset = SignalImplanter._implant_signals(input_params)

        return dataset

    @staticmethod
    def _implant_signals(input_params: dict = None) -> RepertoireDataset:

        PathBuilder.build(input_params["result_path"])

        dataset = input_params["dataset"]
        processed_filenames = []
        simulation_limits = SignalImplanter._prepare_simulation_limits(input_params["simulation"], dataset.get_example_count())
        simulation_index = 0

        for index, repertoire in enumerate(dataset.get_data(input_params["batch_size"])):

            if simulation_index <= len(simulation_limits) - 1 and index >= simulation_limits[simulation_index]:
                simulation_index += 1

            filename = SignalImplanter._process_repertoire(index, repertoire, simulation_index, simulation_limits, input_params)
            processed_filenames.append(filename)

        processed_dataset = RepertoireDataset(filenames=processed_filenames, params=dataset.params,
                                              metadata_file=dataset.metadata_file)
        PickleExporter.export(processed_dataset, input_params["result_path"], FilenameHandler.get_dataset_name(SignalImplanter.__name__))

        return processed_dataset

    @staticmethod
    def _process_repertoire(index, repertoire, simulation_index, simulation_limits, input_params):

        if simulation_index < len(simulation_limits):
            filename = SignalImplanter._implant_in_repertoire(index, repertoire, simulation_index, input_params)
        else:
            filename = SignalImplanter._copy_repertoire(index, repertoire, input_params)

        return filename

    @staticmethod
    def _copy_repertoire(index: int, repertoire: SequenceRepertoire, input_params: dict) -> str:
        new_repertoire = copy.deepcopy(repertoire)
        if new_repertoire.metadata is None:
            new_repertoire.metadata = RepertoireMetadata(sample=Sample(identifier=""))

        # TODO: make it work like in this comment: the user is able to define what a label means (e.g. is it a signal
        #   or a combination of signals, maybe even consider the implanting percentages on the receptor_sequence level?
        # for label in input_params["simulation"]["labels"]:
        #    new_repertoire.metadata.custom_params[label["id"]] = False  # TODO: define the constants somewhere

        for signal in input_params["signals"]:
            new_repertoire.metadata.custom_params[signal.id] = False

        filename = input_params["result_path"] + "rep" + str(index) + ".pickle"

        with open(filename, "wb") as file:
            pickle.dump(new_repertoire, file)

        return filename

    @staticmethod
    def _implant_in_repertoire(index, repertoire, simulation_index, input_params) -> str:
        new_repertoire = repertoire
        for signal in input_params["simulation"][simulation_index]["signals"]:
            new_repertoire = signal.implant_to_repertoire(repertoire=new_repertoire,
                                                          repertoire_implanting_rate=input_params["simulation"][simulation_index]["sequences"])

        for signal in input_params["simulation"][simulation_index]["signals"]:
            new_repertoire.metadata.custom_params[signal.id] = True
        for signal in input_params["signals"]:
            if signal not in input_params["simulation"][simulation_index]["signals"]:
                new_repertoire.metadata.custom_params[signal.id] = False

        filename = input_params["result_path"] + "rep" + str(index) + ".pickle"

        with open(filename, "wb") as file:
            pickle.dump(new_repertoire, file)

        return filename

    @staticmethod
    def _prepare_simulation_limits(simulation: dict, repertoire_count: int) -> list:
        limits = [int(item["repertoires"] * repertoire_count) for item in simulation]
        limits = [sum(limits[:i+1]) for i in range(len(limits))]
        return limits
