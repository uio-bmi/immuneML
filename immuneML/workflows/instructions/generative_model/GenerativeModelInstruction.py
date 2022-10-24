import datetime
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.generative_model.GenerativeModelState import GenerativeModelState
from immuneML.workflows.instructions.generative_model.GenerativeModelUnit import GenerativeModelUnit
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


class GenerativeModelInstruction(Instruction):

    """
    Allows for the generation of data based on existing data

    Arguments:

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
            print(unit)
            print("{}: Started analysis {} ({}/{}).".format(datetime.datetime.now(), key, index+1, len(self.state.generative_model_units)), flush=True)
            path = self.state.result_path / f"analysis_{key}"
            PathBuilder.build(path)
            report_result = self.run_unit(unit, path)
            unit.report_result = report_result
            print("{}: Finished analysis {} ({}/{}).\n".format(datetime.datetime.now(), key, index+1, len(self.state.generative_model_units)), flush=True)
        return self.state

    def run_unit(self, unit: GenerativeModelUnit, result_path: Path) -> ReportResult:
        encoded_dataset = self.encode(unit, result_path / "encoded_dataset")
        unit.report.dataset = encoded_dataset
        unit.genModel.fit(encoded_dataset.encoded_data, unit.label_config.get_label_objects()[0], dataset=unit.dataset)
        unit.genModel.store(result_path)
        unit.genModel.generate()
        unit.report.result_path = result_path / "report"
        report_result = unit.report.generate_report()
        return report_result

    def encode(self, unit: GenerativeModelUnit, result_path: Path) -> Dataset:
        if unit.encoder is not None:
            encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=unit.dataset, encoder=unit.encoder,
                                                                encoder_params=EncoderParams(result_path=result_path,
                                                                                             label_config=unit.label_config,
                                                                                             filename="encoded_dataset.pkl",
                                                                                             learn_model=True,
                                                                                             encode_labels=unit.label_config is not None),
                                                                ))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset
