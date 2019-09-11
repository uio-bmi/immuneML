from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.workflows.processes.InstructionProcess import InstructionProcess
from source.workflows.processes.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams


class ExploratoryAnalysisProcess(InstructionProcess):
    """
    Allows exploratory analysis of different datasets using encodings and reports.

    Analysis is defined by a list of ExploratoryAnalysisUnit objects that encapsulate a dataset, an encoding and a report to be
    executed on the (encoded) dataset.
    """

    def __init__(self, exploratory_analysis_units: list):
        assert all(isinstance(unit, ExploratoryAnalysisUnit) for unit in exploratory_analysis_units), \
            "ExploratoryAnalysisProcess: not all elements passed to init method are instances of ExploratoryAnalysisUnit."

        self.exploratory_analysis_units = exploratory_analysis_units

    def run(self, result_path: str):
        for index, unit in enumerate(self.exploratory_analysis_units):
            print("Started analysis {}/{}.".format(index+1, len(self.exploratory_analysis_units)))
            self.run_unit(unit, result_path + "analysis_{}/".format(index+1))
            print("Finished analysis {}/{}.".format(index+1, len(self.exploratory_analysis_units)))

    def run_unit(self, unit: ExploratoryAnalysisUnit, result_path: str):
        unit.dataset = self.preprocess_dataset(unit, result_path)
        encoded_dataset = self.encode(unit, result_path)
        unit.report.dataset = encoded_dataset
        unit.report.result_path = result_path
        unit.report.generate_report()

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
                                                                                             filename="encoded_dataset.pkl")))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset
