How to add a new report
========================

In this tutorial, we will show how to add a new report to plot sequence length distribution in repertoire datasets. We will call this report
MySequenceLengthDistribution.

Adding a new report
---------------------

To add a new report, add a new class called MySequenceLengthDistribution to the :py:mod:`~immuneML.reports.data_reports` package.

MySequenceLengthDistribution class should inherit :py:mod:`~immuneML.reports.data_reports.DataReport.DataReport` class and implement all abstract methods.

An example implementation is shown below. It includes implementations of abstract methods :code:`build_object(**kwargs)`, :code:`check_prerequisites()` and
:code:`_generate()`, and class documentation at the beginning. This class documentation will be shown to the user.

.. code-block:: python

    import logging
    from collections import Counter
    from pathlib import Path

    import pandas as pd
    import plotly.express as px

    from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
    from immuneML.data_model.repertoire.Repertoire import Repertoire
    from immuneML.reports.ReportOutput import ReportOutput
    from immuneML.reports.ReportResult import ReportResult
    from immuneML.reports.data_reports.DataReport import DataReport
    from immuneML.util.PathBuilder import PathBuilder


    class SequenceLengthDistribution(DataReport):
        """
        Generates a histogram of the lengths of the sequences in a RepertoireDataset.

        YAML specification:

        .. indent with spaces
        .. code-block:: yaml

            my_sld_report: SequenceLengthDistribution

        """

        @classmethod
        def build_object(cls, **kwargs): # called when parsing YAML - all checks for parameters (if any) should be in this function
            return SequenceLengthDistribution(**kwargs)

        def __init__(self, dataset: RepertoireDataset = None, batch_size: int = 1, result_path: Path = None, name: str = None):
            super().__init__(dataset=dataset, result_path=result_path, name=name)
            self.batch_size = batch_size

        def check_prerequisites(self): # called at runtime to check if the report can be run with params assigned at runtime (e.g., dataset is set at runtime)
            if isinstance(self.dataset, RepertoireDataset):
                return True
            else:
                logging.warning("SequenceLengthDistribution: report can be generated only from RepertoireDataset. Skipping this report...")
                return False

        def _generate(self) -> ReportResult: # the function that creates the report
            sequence_lengths = self._get_sequence_lengths()
            report_output_fig = self._plot(sequence_lengths=sequence_lengths)
            output_figures = None if report_output_fig is None else [report_output_fig]
            return ReportResult(type(self).__name__, output_figures=output_figures)

        def _get_sequence_lengths(self) -> Counter: # implementation detail: extract sequence lengths from repertoires in the dataset
            sequence_lenghts = Counter()

            for repertoire in self.dataset.get_data(self.batch_size):
                seq_lengths = self._count_in_repertoire(repertoire)
                sequence_lenghts += seq_lengths

            return sequence_lenghts

        def _count_in_repertoire(self, repertoire: Repertoire) -> Counter: # implementation detail: get lengths of sequences for one repertoire
            c = Counter([len(sequence.get_sequence()) for sequence in repertoire.sequences])
            return c

        def _plot(self, sequence_lengths: Counter) -> ReportOutput: # implementation detail: when all lengths are know, plot them

            df = pd.DataFrame({"counts": list(sequence_lengths.values()), 'sequence_lengths': list(sequence_lengths.keys())})

            figure = px.bar(df, x="sequence_lengths", y="counts")
            figure.update_layout(xaxis=dict(tickmode='array', tickvals=df["sequence_lengths"]), yaxis=dict(tickmode='array', tickvals=df["counts"]),
                                 title="Sequence length distribution", template="plotly_white")
            figure.update_traces(marker_color=px.colors.diverging.Tealrose[0])
            PathBuilder.build(self.result_path)

            file_path = self.result_path / "sequence_length_distribution.html"
            figure.write_html(str(file_path))
            return ReportOutput(path=file_path, name="sequence length distribution plot")


Unit testing the new report
----------------------------

To add a unit test:

1. Add a new file to :py:mod:`~test.reports.data_reports` package named test_mySequenceLengthDistribution.py.

2. Add a class TestMySequenceLengthDistribution that inherits :code:`unittest.TestCase` to the new file.

3. Add a function to set up cache used for testing.

4. Define tests for functions you implemented.

Typically, the :code:`generate_report()` function of the new report should be tested, as well as other relevant methods, to ensure that the report output is correct.
When building tests for reports, a useful class is :py:obj:`~immuneML.simulation.dataset_generation.RandomDatasetGenerator.RandomDatasetGenerator`, which can create a dataset with random sequences.

An example of the test is given below.

.. code-block:: python

    import os
    import shutil
    from unittest import TestCase

    from immuneML.caching.CacheType import CacheType
    from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
    from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
    from immuneML.data_model.repertoire.Repertoire import Repertoire
    from immuneML.environment.Constants import Constants
    from immuneML.environment.EnvironmentSettings import EnvironmentSettings
    from immuneML.reports.data_reports.MySequenceLengthDistribution import MySequenceLengthDistribution
    from immuneML.util.PathBuilder import PathBuilder


    class TestSequenceLengthDistribution(TestCase):

        def setUp(self) -> None:
            os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

        def test_generate_report(self):
            path = EnvironmentSettings.root_path / "test/tmp/datareports/"
            PathBuilder.build(path)

            rep1 = Repertoire.build_from_sequence_objects(sequence_objects=[ReceptorSequence(amino_acid_sequence="AAA", identifier="1"),
                                                                            ReceptorSequence(amino_acid_sequence="AAAA", identifier="2"),
                                                                            ReceptorSequence(amino_acid_sequence="AAAAA", identifier="3"),
                                                                            ReceptorSequence(amino_acid_sequence="AAA", identifier="4")],
                                                          path=path, metadata={})
            rep2 = Repertoire.build_from_sequence_objects(sequence_objects=[ReceptorSequence(amino_acid_sequence="AAA", identifier="5"),
                                                                            ReceptorSequence(amino_acid_sequence="AAAA", identifier="6"),
                                                                            ReceptorSequence(amino_acid_sequence="AAAA", identifier="7"),
                                                                            ReceptorSequence(amino_acid_sequence="AAA", identifier="8")],
                                                          path=path, metadata={})

            dataset = RepertoireDataset(repertoires=[rep1, rep2])

            report = MySequenceLengthDistribution(dataset, 1, path)

            result = report.generate_report()
            self.assertTrue(os.path.isfile(result.output_figures[0].path))

            shutil.rmtree(path)


Adding a report: additional information
----------------------------------------

In immuneML, it is possible to automatically generate a report describing some aspect of the problem being examined. There are a few types of reports:

  #. Data report – reports examining some aspect of the dataset (such as sequence length distribution, gene usage)
  #. Encoding report – shows some aspect of the encoded dataset (such as the feature values of an encoded dataset),
  #. ML model report – shows the characteristics of an inferred machine learning model (such as coefficient values for logistic regression or kernel visualization for CNN)
  #. Train ML model report – show statistics of multiple trained ML models in the TrainMLModelInstruction (such as comparing performance statistics between models, or performance w.r.t. an encoding parameter)
  #. Multi dataset report –  show statistics when running immuneML with the MultiDatasetBenchmarkTool

These types of reports are modeled by the following classes:

  #. :py:obj:`~immuneML.reports.data_reports.DataReport.DataReport`
  #. :py:obj:`~immuneML.reports.encoding_reports.EncodingReport.EncodingReport`
  #. :py:obj:`~immuneML.reports.ml_reports.MLReport.MLReport`
  #. :py:obj:`~immuneML.reports.train_ml_model_reports.TrainMLModelReport.TrainMLModelReport`
  #. :py:obj:`~immuneML.reports.multi_dataset_reports.MultiDatasetReport.MultiDatasetReport`

The existing reports can be found in the package :py:mod:`~immuneML.reports`. These can be specified in the YAML by specifying the name and optional parameters
(see: :ref:`How to specify an analysis with YAML`).

Creating a custom report
**************************

Determine the type of report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, it is important to determine what the type of the report is, as this defines which report class should be inherited.

If the report will be used to analyze a Dataset (such as a RepertoireDataset), either a DataReport or an EncodingReport should be used. The simplest
report is the DataReport, which should typically be used when summarizing some qualities of a dataset. This dataset can be found in the report
attribute dataset.

Use the EncodingReport when it is necessary to access the encoded_data attribute of a `Dataset`. The encoded_data attribute is an instance of a
:py:obj:`~immuneML.data_model.encoded_data.EncodedData.EncodedData` class. This report should be used when the data
representation first needs to be changed before running the report, either through an existing or a custom encoding (see:
:ref:`How to add a new encoding`). For example, the :ref:`Matches` report represents a RepertoireDataset based on matches to a given reference
dataset, and must first be encoded using a :ref:`MatchedSequences`, :ref:`MatchedReceptors` or :ref:`MatchedRegex`.

When the results of an experiment with a machine learning method should be analyzed, an MLReport or TrainMLModelReport should be used. These reports
are a bit more advanced and require more understanding of the TrainMLModelInstruction. The MLReport should be used when plotting statistics or
exporting information about one trained ML model. This report can be executed on any trained ML model, both in the assessment and selection loop of
the :ref:`TrainMLModel`. An MLReport has the following attributes:

  #. train_dataset: a Dataset (e.g., RepertoireDataset) object containing the training data used for the given classifier
  #. test_dataset: similar to train_dataset, but containing the test data
  #. method: the MLMethod object containing trained classifiers for each of the labels.
  #. label: the label that the report is executed for (the same report may be executed several times when training classifiers for multiple labels), can be used to retrieve relevant information from the MLMethod object.
  #. hp_setting: the :py:obj:`~immuneML.hyperparameter_optimization.HPSetting.HPSetting` object, containing all information about which preprocessing, encoding, and ML methods were used up to this point. This parameter can usually be ignored unless specific information from previous analysis steps is needed.

In contrast, `TrainMLModelReport` is used to compare several [optimal] ML models. This report has access to the attribute state: a :py:obj:`~immuneML.hyperparameter_optimization.states.TrainMLModelState.TrainMLModelState`
object, containing information that has been collected through the execution of the TrainMLModelInstruction. This includes all datasets, trained
models, labels, internal state objects for selection and assessment loops (nested cross-validation), optimal models, and more.

Finally, the MultiDatasetReport is used in rare cases when running immuneML with the MultiDatasetBenchmarkTool. This can be used when comparing the
performance of classifiers over several datasets and accumulating the results. This report has the attribute instruction_states: a list of several
TrainMLModelState objects.

Implementing the report
^^^^^^^^^^^^^^^^^^^^^^^^

The new report should inherit the appropriate report type and be placed in the respective package (under :py:mod:`~immuneML.reports`, choose :code:`data_reports`,
:code:`encoding_reports`, :code:`ml_reports`, :code:`train_ml_model_reports`, or :code:`multidataset_reports`). The abstract method `generate()` must be implemented,
which has the following responsibilities:

  - It should create the report results, for example, compute the data or create the plots that should be returned by the report.
  - It should write the report results to the folder given at the variable result_path.
  - It should return a ReportResult object, which contains lists of ReportOutput objects. These ReportOutput objects simply contain the path to a figure, table, text, or another type of result. One report can have multiple outputs, as long as they are all accessible through the ReportResult. This will be later used to format the summary of the results in the HTML output file.
  - When the main result of the report is a plot, it is good practice to also make the raw data available to the user, for example as a csv file.

The preferred method for plotting data is through `plotly <https://plotly.com/python/>`_, as it creates interactive and rescalable plots in HTML format [recommended] that
display nicely in the HTML output file. Alternatively, plots can also be in pdf, png, jpg and svg format.

.. note::

    When plotting data with `plotly <https://plotly.com/python/>`_, we recommend using the following color schemes for consistency:
    plotly.colors.sequential.Teal, plotly.colors.sequential.Viridis, or plotly.colors.diverging.Tealrose.
    Additionally, in the most of immuneML plots, 'plotly_white' theme is used for the background.

    For the overview of color schemes, visit `this link <https://plotly.com/python/builtin-colorscales/>`_.
    For plotly themes, visit `this link <https://plotly.com/python/templates/>`_.

The second abstract method to be implemented is `build_object()`. This method can take in any custom parameters and should return an instance of the
report object. The parameters of the method `build_object()` can be directly specified in the YAML specification, nested under the report type, for example:

.. code-block:: yaml

  MyNewReport:
    custom_parameter: “value”


Inside the :code:`build_object()` method, you can check if the correct parameters are specified and raise an exception when the user input is incorrect
(for example using the :py:obj:`~immuneML.util.ParameterValidator.ParameterValidator` utility class). Furthermore, it is possible to resolve more
complex input parameters, such as loading reference sequences from an external input file, before passing them to the :code:`__init__()` method of the report.

It is important to consider whether the method :code:`check_prerequisites()` should be implemented. This method should return a boolean value describing
whether the prerequisites are met, and print a warning message to the user when this condition is false. The report will only be generated when
:code:`check_prerequisites()` returns true. This method should not be used to raise exceptions. Instead, it is used to prevent exceptions from happening
during execution, as this might cause lost results. Situations to consider are:

  - When implementing an EncodingReport, use this function to check that the data has been encoded and that the correct encoder has been used.
  - Similarly, when creating an MLReport or TrainMLModelReport, check that the appropriate ML methods have been used.

.. include:: ./dev_docs_util.rst

.. note::

  Please see the :py:obj:`~immuneML.reports.Report.Report` class for the detailed description of the methods to be implemented.

Test run of the report: specifying the new report in YAML
***********************************************************

Custom reports may be defined in the YAML specification under the key ‘definitions’ the same way as any other reports. The easiest way to test run
`Data reports <https://docs.immuneml.uio.no/specification.html#data-reports>`_ and `Encoding reports <https://docs.immuneml.uio.no/specification.html#encoding-reports>`_ is through the :ref:`ExploratoryAnalysis` instruction. They may also be specified in the :ref:`TrainMLModel`
instruction in the :code:`selection` and :code:`assessment` loop under :code:`reports:data_splits` and :code:`reports:encoding` respectively.

`ML model reports <https://docs.immuneml.uio.no/specification.html#ml-model-reports>`_ and `Train ML model reports <https://docs.immuneml.uio.no/specification.html#train-ml-model-reports>`_ can only be run through the :ref:`TrainMLModel` instruction. :ref:`ML reports` can be specified inside both the
:code:`selection` and :code:`assessment` loop under :code:`reports:models`. :ref:`Train ML model reports` must be specified under :code:`reports`.

Finally, :ref:`Multi dataset reports` can be specified under :code:`benchmark_reports` when running the :ref:`MultiDatasetBenchmarkTool`.

The following specification shows the places where `Data reports <https://docs.immuneml.uio.no/specification.html#data-reports>`_,
`Encoding reports <https://docs.immuneml.uio.no/specification.html#encoding-reports>`_ ,
`ML model reports <https://docs.immuneml.uio.no/specification.html#ml-model-reports>`_,
and `Train ML model reports <https://docs.immuneml.uio.no/specification.html#train-ml-model-reports>`_ can be specified:

.. code-block:: yaml

  definitions:
    reports:
      my_data_report: MyNewDataReport # example data report without parameters
      my_encoding_report: # example encoding report with a parameter
        MyNewEncodingReport:
         parameter: value
      my_ml_report: MyNewMLReport # ml model report
      my_trainml_report: MyNewTrainMLModelReport # train ml model report

    datasets:
      d1:
        # if you do not have real data to test your report with, consider
        # using a randomly generated dataset, see the documentation:
        # “How to generate a random receptor or repertoire dataset”
        format: RandomRepertoireDataset
        params:
            labels: {disease: {True: 0.5, False: 0.5}}
            repertoire_count: 50
    encodings:
      e1: KmerFrequency
    ml_methods:
      m1: LogisticRegression

  instructions:
    exploratory_instr: # Example of specifying reports in ExploratoryAnalysis
      type: ExploratoryAnalysis
      analyses:
        analysis_1: # Example analysis with data report
          dataset: d1
          report: my_data_report
        analysis_1: # Example analysis with encoding report
          dataset: d1
          encoding: e1
          report: my_encoding_report
          labels: # when running an encoding report, labels must be specified
              - disease

    trainmlmodel_instr: # Example of specifying reports in TrainMLModel instruction
      type: TrainMLModel
      settings:
        - encoding: e1
          ml_method: m1
      assessment: # running reports in the assessment (outer) loop
        reports:
          data: # execute before splitting to training/(validation+test)
            - my_data_report
          data_splits: # execute on training and (validation+test) sets
            - my_data_report
          encoding:
            - my_encoding_report
          models:
            - my_ml_report
      selection: # running reports in the selection (inner) loop
        reports:
          data: # execute before splitting to validation/test
            - my_data_report
          data_splits: # execute on validation and test sets
            - my_data_report
          encoding:
            - my_encoding_report
          models:
            - my_ml_report
      reports:
        - my_trainml_report
      labels:
        - disease

Adding documentation for the new report
****************************************************

After implementing the desired functionality, the documentation for the report should be added, so that the users of immuneML have sufficient information when deciding to use the report. It should be added to the docstring and consist of the following components:

  #. A short description of what the report is meant for.

  #. Optional extended description, including any references or specific cases that should bee considered.

  #. List of arguments the report takes as input. If the report does not take any arguments other than the ones provided by the immuneML in runtime depending on the report type (such as training and test dataset or trained method), there should be only a short statement that the report does not take input arguments.

  #. An example of how the report can be specified in YAML.

Here is an example of a documentation for the :ref:`DesignMatrixExporter` report that has no input arguments which can be provided by the user in the YAML
specification (the encoded dataset to be exported will be provided by immuneML at runtime):

.. code-block:: text

    Generates a histogram of the lengths of the sequences in a RepertoireDataset.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_sld_report: SequenceLengthDistribution