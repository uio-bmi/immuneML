from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.reports.ReportResult import ReportResult
from source.workflows.instructions.Instruction import Instruction
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams


class ExploratoryAnalysisInstruction(Instruction):
    """
    Allows exploratory analysis of different datasets using encodings and reports.

    Analysis is defined by a list of ExploratoryAnalysisUnit objects that encapsulate a dataset, an encoding and a report to be
    executed on the (encoded) dataset.
    """

    def __init__(self, exploratory_analysis_units: dict, name: str = None):
        assert all(isinstance(unit, ExploratoryAnalysisUnit) for unit in exploratory_analysis_units.values()), \
            "ExploratoryAnalysisInstruction: not all elements passed to init method are instances of ExploratoryAnalysisUnit."
        self.state = ExploratoryAnalysisState(exploratory_analysis_units, name)

    def run(self, result_path: str):
        self.state.result_path = result_path
        for index, (key, unit) in enumerate(self.state.exploratory_analysis_units.items()):
            print("Started analysis {}/{}.".format(index+1, len(self.state.exploratory_analysis_units)))
            report_result = self.run_unit(unit, result_path + "analysis_{}/".format(key))
            unit.report_result = report_result
            print("Finished analysis {}/{}.".format(index+1, len(self.state.exploratory_analysis_units)))
        return self.state

    def run_unit(self, unit: ExploratoryAnalysisUnit, result_path: str) -> ReportResult:
        unit.dataset = self.preprocess_dataset(unit, result_path)
        encoded_dataset = self.encode(unit, result_path)
        unit.report.dataset = encoded_dataset
        unit.report.result_path = result_path
        report_result = unit.report.generate_report()
        return report_result

    def preprocess_dataset(self, unit: ExploratoryAnalysisUnit, result_path: str) -> Dataset:
        if unit.preprocessing_sequence is not None and len(unit.preprocessing_sequence) > 0:
            dataset = unit.dataset
            for preprocessing in unit.preprocessing_sequence:
                dataset = preprocessing.process_dataset(dataset, result_path)
        else:
            dataset = unit.dataset
        return dataset

    def encode(self, unit: ExploratoryAnalysisUnit, result_path: str) -> Dataset:
        if unit.encoder is not None:
            encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=unit.dataset, encoder=unit.encoder,
                                                                encoder_params=EncoderParams(result_path=result_path,
                                                                                             label_configuration=unit.label_config,
                                                                                             filename="encoded_dataset.pkl",
                                                                                             batch_size=unit.batch_size)))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset
