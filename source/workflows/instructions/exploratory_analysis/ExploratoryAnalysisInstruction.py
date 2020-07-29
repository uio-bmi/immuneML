from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.reports.ReportResult import ReportResult
from source.util.PathBuilder import PathBuilder
from source.workflows.instructions.Instruction import Instruction
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState
from source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataEncoderParams import DataEncoderParams


class ExploratoryAnalysisInstruction(Instruction):
    """
    Allows exploratory analysis of different datasets using encodings and reports.

    Analysis is defined by a dictionary of ExploratoryAnalysisUnit objects that encapsulate a dataset, an encoding [optional]
    and a report to be executed on the [encoded] dataset. Each analysis in specified under `analyses` is completely independent from all
    others.

    Arguments:

        exploratory_analysis_units (dict): a dictionary of analyses to perform; keys are the names of the analyses and values are the parameters of
        the analyses (objects of :py:obj:`~source.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit.ExploratoryAnalysisUnit`
        class). Each of these includes a dataset on which to perform the analysis, report to run, and optionally preprocessing sequence,
        encoding (if the report needs to be executed on the encoded dataset) and label configuration (if the dataset needs to be encoded,
        it is also necessary to specify a label for encoding - a label could correspond to an immune event or to genetic information (e.g. HLA).
        The batch size can also be set to encode multiple repertoires in parallel, but this also increases the memory usage, so it must be
        reasonably set.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_dataset_generation_instruction: # user-defined instruction name
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
                    batch_size: 16

    """

    def __init__(self, exploratory_analysis_units: dict, name: str = None):
        assert all(isinstance(unit, ExploratoryAnalysisUnit) for unit in exploratory_analysis_units.values()), \
            "ExploratoryAnalysisInstruction: not all elements passed to init method are instances of ExploratoryAnalysisUnit."
        self.state = ExploratoryAnalysisState(exploratory_analysis_units, name=name)
        self.name = name

    def run(self, result_path: str):
        self.state.result_path = result_path + f"{self.name}/"
        for index, (key, unit) in enumerate(self.state.exploratory_analysis_units.items()):
            print("Started analysis {}/{}.".format(index+1, len(self.state.exploratory_analysis_units)))
            path = self.state.result_path + "analysis_{}/".format(key)
            PathBuilder.build(path)
            report_result = self.run_unit(unit, path)
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
