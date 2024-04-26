from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisState import ExploratoryAnalysisState
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams
from immuneML.workflows.steps.DataWeighter import DataWeighter
from immuneML.workflows.steps.DataWeighterParams import DataWeighterParams


class ExploratoryAnalysisInstruction(Instruction):
    """
    Allows exploratory analysis of different datasets using encodings and reports.

    Analysis is defined by a dictionary of ExploratoryAnalysisUnit objects that encapsulate a dataset, an encoding [optional]
    and a report to be executed on the [encoded] dataset. Each analysis specified under `analyses` is completely independent from all
    others.

    **Specification arguments:**

    - analyses (dict): a dictionary of analyses to perform. The keys are the names of different analyses, and the values for each
      of the analyses are:

      - dataset: dataset on which to perform the exploratory analysis

      - preprocessing_sequence: which preprocessings to use on the dataset, this item is optional and does not have to be specified.

      - example_weighting: which example weighting strategy to use before encoding the data, this item is optional and does not have to be specified.

      - encoding: how to encode the dataset before running the report, this item is optional and does not have to be specified.

      - labels: if encoding is specified, the relevant labels should be specified here.

      - dim_reduction: which dimensionality reduction to apply; this is an experimental feature

      - report: which report to run on the dataset. Reports specified here may be of the category :ref:`Data reports` or :ref:`Encoding reports`, depending on whether 'encoding' was specified.

    - number_of_processes: (int): how many processes should be created at once to speed up the analysis. For personal
      machines, 4 or 8 is usually a good choice.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        instructions:
            my_expl_analysis_instruction: # user-defined instruction name
                type: ExploratoryAnalysis # which instruction to execute
                analyses: # analyses to perform
                    my_first_analysis: # user-defined name of the analysis
                        dataset: d1 # dataset to use in the first analysis
                        preprocessing_sequence: p1 # preprocessing sequence to use in the first analysis
                        report: r1 # which report to generate using the dataset d1
                    my_second_analysis: # user-defined name of another analysis
                        dataset: d1 # dataset to use in the second analysis - can be the same or different from other analyses
                        encoding: e1 # encoding to apply on the specified dataset (d1)
                        report: r2 # which report to generate in the second analysis
                        labels: # labels present in the dataset d1 which will be included in the encoded data on which report r2 will be run
                            - celiac # name of the first label as present in the column of dataset's metadata file
                            - CMV # name of the second label as present in the column of dataset's metadata file
                    my_third_analysis: # user-defined name of another analysis
                        dataset: d1 # dataset to use in the second analysis - can be the same or different from other analyses
                        encoding: e1 # encoding to apply on the specified dataset (d1)
                        dim_reduction: umap # or None; which dimensionality reduction method to apply to encoded d1
                        report: r3 # which report to generate in the third analysis
                number_of_processes: 4 # number of parallel processes to create (could speed up the computation)
    """

    def __init__(self, exploratory_analysis_units: dict, name: str = None):
        assert all(isinstance(unit, ExploratoryAnalysisUnit) for unit in exploratory_analysis_units.values()), \
            ("ExploratoryAnalysisInstruction: not all elements passed to init method are instances of "
             "ExploratoryAnalysisUnit.")
        self.state = ExploratoryAnalysisState(exploratory_analysis_units, name=name)
        self.name = name

    def run(self, result_path: Path):
        name = self.name if self.name is not None else "exploratory_analysis"
        self.state.result_path = result_path / name
        for index, (key, unit) in enumerate(self.state.exploratory_analysis_units.items()):
            print_log(f"Started analysis {key} ({index+1}/{len(self.state.exploratory_analysis_units)}).",
                      include_datetime=True)
            path = self.state.result_path / f"analysis_{key}"
            PathBuilder.build(path)
            report_result = self.run_unit(unit, path)
            unit.report_result = report_result
            print_log(f"Finished analysis {key} ({index+1}/{len(self.state.exploratory_analysis_units)}).\n",
                      include_datetime=True)
        return self.state

    def run_unit(self, unit: ExploratoryAnalysisUnit, result_path: Path) -> ReportResult:
        unit.dataset = self.preprocess_dataset(unit, result_path / "preprocessed_dataset")
        unit.dataset = self.weight_examples(unit, result_path / "weighted_dataset")
        unit.dataset = self.encode(unit, result_path / "encoded_dataset")

        if unit.dim_reduction is not None:
            self._run_dimensionality_reduction(unit)

        if unit.report is not None:
            report_result = self.run_report(unit, result_path)
        else:
            report_result = None

        return report_result

    def _run_dimensionality_reduction(self, unit: ExploratoryAnalysisUnit):
        result = unit.dim_reduction.fit_transform(unit.dataset)
        unit.dataset.encoded_data.dimensionality_reduced_data = result

    def preprocess_dataset(self, unit: ExploratoryAnalysisUnit, result_path: Path) -> Dataset:
        if unit.preprocessing_sequence is not None and len(unit.preprocessing_sequence) > 0:
            dataset = unit.dataset
            for preprocessing in unit.preprocessing_sequence:
                dataset = preprocessing.process_dataset(dataset, result_path)
        else:
            dataset = unit.dataset
        return dataset

    def weight_examples(self, unit: ExploratoryAnalysisUnit, result_path: Path):
        if unit.example_weighting is not None:
            weighted_dataset = DataWeighter.run(DataWeighterParams(dataset=unit.dataset, weighting_strategy=unit.example_weighting,
                                                                   weighting_params=ExampleWeightingParams(result_path=result_path,
                                                                                                           pool_size=unit.number_of_processes,
                                                                                                           learn_model=True),
                                                                   ))
        else:
            weighted_dataset = unit.dataset

        return weighted_dataset

    def encode(self, unit: ExploratoryAnalysisUnit, result_path: Path) -> Dataset:
        if unit.encoder is not None:
            encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=unit.dataset, encoder=unit.encoder,
                                                                encoder_params=EncoderParams(result_path=result_path,
                                                                                             label_config=unit.label_config,
                                                                                             pool_size=unit.number_of_processes,
                                                                                             learn_model=True,
                                                                                             encode_labels=unit.label_config is not None),
                                                                ))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset

    def run_report(self, unit: ExploratoryAnalysisUnit, result_path: Path):
        unit.report.result_path = result_path / "report"
        unit.report.number_of_processes = unit.number_of_processes

        unit.report.dataset = unit.dataset
        return unit.report.generate_report()

