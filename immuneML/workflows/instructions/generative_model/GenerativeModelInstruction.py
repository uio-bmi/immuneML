import datetime
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.generative_model.GenerativeModelState import GenerativeModelState
from immuneML.workflows.instructions.generative_model.GenerativeModelUnit import GenerativeModelUnit
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams
from immuneML.environment.LabelConfiguration import LabelConfiguration


class GenerativeModelInstruction(Instruction):

    """
    Allows for the generation of data based on existing data

    Analysis is defined by a dictionary of GenerativeModelUnits objects that encapsulate a Generative method, a dataset,
    an encoding and a report to be executed on the [encoded] dataset. Each generator specified in the dictionary is
    completely independent of all others.

    Arguments:

        generative_model_units (dict): a dictionary of generators to execute. The keys are the names of different
        generators, and the values for each of the generators are:

        - GenerativeModel: a subclass of the GenerativeModel class, providing training and generation methods
        - dataset: dataset on which to train the generators
        - encoding: how to encode the dataset before running the report
        - report: which report to run on the dataset. Reports specified here may be of the category :ref:`Data reports`,
            :ref:'ML reports, or :ref:`Encoding reports`, depending on what the user wishes to report.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_generative_model_instruction: # user-defined instruction name
            type: GenerativeModel # which instruction to execute
            generators: # generators to execute
                my_first_generator: # user-defined name of the generator
                    ml_method: m1 # ml_method specified by the user, must be a subclass of GenerativeModel
                    dataset: d1 # dataset to use in the first generator
                    encoding: e1 # encoding to apply to dataset (d1)
                    report: r1 # which report to generate using the dataset d1
                my_second_generator: # user-defined name of another generator
                    ml_method: m2 # ml_method specified by the user, must be a subclass of GenerativeModel
                    dataset: d2 # dataset to use in the second generator - can be the same or different from other generators
                    encoding: e2 # encoding to apply on the specified dataset (d2)
                    report: r2 # which report to generate in the second generator

    """

    def __init__(self, generative_model_units: dict, name: str = None):
        assert all(isinstance(unit, GenerativeModelUnit) for unit in generative_model_units.values()), \
            "GenerativeModelInstruction: not all elements passed to init method are instances of GenerativeModelUnit."
        self.state = GenerativeModelState(generative_model_units, name=name)
        self.name = name

    def run(self, result_path: Path):
        name = self.name if self.name is not None else "generative_model"
        self.state.result_path = result_path / name
        for index, (key, unit) in enumerate(self.state.generative_model_units.items()):
            print("{}: Started generator {} ({}/{}).".format(datetime.datetime.now(), key, index+1, len(self.state.generative_model_units)), flush=True)
            path = self.state.result_path / f"analysis_{key}"
            PathBuilder.build(path)
            report_result = self.run_unit(unit, path)
            unit.report_result = report_result
            print("{}: Finished analysis {} ({}/{}).\n".format(datetime.datetime.now(), key, index+1, len(self.state.generative_model_units)), flush=True)
        return self.state

    def run_unit(self, unit: GenerativeModelUnit, result_path: Path) -> ReportResult:
        encoded_dataset = self.encode(unit, result_path / "encoded_dataset")
        unit.report.dataset = encoded_dataset
        unit.genModel.fit(encoded_dataset.encoded_data)
        print_log(f"Finished training", include_datetime=True)
        unit.genModel.store(result_path)
        unit.generated_sequences = unit.genModel.generate()
        print_log(f"Finished generation", include_datetime=True)
        unit.report.sequences = unit.generated_sequences
        unit.report.alphabet = unit.genModel.alphabet
        unit.report.method = unit.genModel
        unit.report.result_path = result_path / "report"
        report_result = unit.report.generate_report()
        return report_result

    def encode(self, unit: GenerativeModelUnit, result_path: Path) -> Dataset:
        if unit.encoder is not None:
            encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=unit.dataset, encoder=unit.encoder,
                                                                encoder_params=EncoderParams(result_path=result_path,
                                                                                             filename="encoded_dataset.pkl",
                                                                                             label_config=None,
                                                                                             encode_labels=False,
                                                                                             learn_model=True),
                                                                ))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset
