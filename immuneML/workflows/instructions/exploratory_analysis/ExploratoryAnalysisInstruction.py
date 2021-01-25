import datetime
from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams


class ExploratoryAnalysisInstruction(Instruction):
    """
    Allows exploratory analysis of different datasets using encodings and reports.

    Analysis is defined by a dictionary of ExploratoryAnalysisUnit objects that encapsulate a dataset, an encoding [optional]
    and a report to be executed on the [encoded] dataset. Each analysis specified under `analyses` is completely independent from all
    others.

    Arguments:

        analyses (dict): a dictionary of analyses to perform. The keys are the names of different analyses, and the values for each
        of the analyses are:

        - dataset: dataset on which to perform the exploratory analysis
        - preprocessing_sequence: which preprocessings to use on the dataset, this item is optional and does not have to be specified.
        - encoding: how to encode the dataset before running the report, this item is optional and does not have to be specified.
        - labels: if encoding is specified, the relevant labels must be specified here.
        - report: which report to run on the dataset. Reports specified here may be of the category :ref:`Data reports` or :ref:`Encoding reports`, depending on whether 'encoding' was specified.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_expl_analysis_instruction: # user-defined instruction name
            type: ExploratoryAnalysis # which instruction to execute
            analyses: # analyses to perform
                my_first_analysis: # user-defined name of the analysis
                    dataset: d1 # dataset to use in the first analysis
                    report: r1 # which report to generate using the dataset d1
                my_second_analysis: # user-defined name of another analysis
                    dataset: d1 # dataset to use in the second analysis - can be the same or different from other analyses
                    encoding: e1 # encoding to apply on the specified dataset (d1)
                    report: r2 # which report to generate in the second analysis
                    labels: # labels present in the dataset d1 which will be included in the encoded data on which report r2 will be run
                        - celiac # name of the first label as present in the column of dataset's metadata file
                        - CMV # name of the second label as present in the column of dataset's metadata file

    """

    def __init__(self, exploratory_analysis_units: dict, name: str = None):
        assert all(isinstance(unit, ExploratoryAnalysisUnit) for unit in exploratory_analysis_units.values()), \
            "ExploratoryAnalysisInstruction: not all elements passed to init method are instances of ExploratoryAnalysisUnit."
        self.state = ExploratoryAnalysisState(exploratory_analysis_units, name=name)
        self.name = name

    def run(self, result_path: Path):
        name = self.name if self.name is not None else "exploratory_analysis"
        self.state.result_path = result_path / name
        for index, (key, unit) in enumerate(self.state.exploratory_analysis_units.items()):
            print("{}: Started analysis {} ({}/{}).".format(datetime.datetime.now(), key, index+1, len(self.state.exploratory_analysis_units)), flush=True)
            path = self.state.result_path / f"analysis_{key}"
            PathBuilder.build(path)
            report_result = self.run_unit(unit, path)
            unit.report_result = report_result
            print("{}: Finished analysis {} ({}/{}).\n".format(datetime.datetime.now(), key, index+1, len(self.state.exploratory_analysis_units)), flush=True)
        return self.state

    def run_unit(self, unit: ExploratoryAnalysisUnit, result_path: Path) -> ReportResult:
        unit.dataset = self.preprocess_dataset(unit, result_path / "preprocessed_dataset")
        encoded_dataset = self.encode(unit, result_path / "encoded_dataset")
        unit.report.dataset = encoded_dataset
        unit.report.result_path = result_path / "report"
        report_result = unit.report.generate_report()
        return report_result

    def preprocess_dataset(self, unit: ExploratoryAnalysisUnit, result_path: Path) -> Dataset:
        if unit.preprocessing_sequence is not None and len(unit.preprocessing_sequence) > 0:
            dataset = unit.dataset
            for preprocessing in unit.preprocessing_sequence:
                dataset = preprocessing.process_dataset(dataset, result_path)
        else:
            dataset = unit.dataset
        return dataset

    def encode(self, unit: ExploratoryAnalysisUnit, result_path: Path) -> Dataset:
        if unit.encoder is not None:
            encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=unit.dataset, encoder=unit.encoder,
                                                                encoder_params=EncoderParams(result_path=result_path,
                                                                                             label_config=unit.label_config,
                                                                                             filename="encoded_dataset.pkl",
                                                                                             pool_size=unit.number_of_processes),
                                                                store_encoded_data=True))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset
