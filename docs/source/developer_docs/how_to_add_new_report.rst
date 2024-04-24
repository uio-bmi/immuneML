How to add a new report
========================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML dev docs: add a new analysis report
   :twitter:description: See how to add a new report to the immuneML platform.
   :twitter:image: https://docs.immuneml.uio.no/_images/extending_immuneML.png


Adding an example data report to the immuneML codebase
-----------------------------------------------------------

In this tutorial, we will show how to add a new report to plot sequence length distribution in repertoire datasets.
This tutorial assumes you have installed immuneML for development as described at :ref:`Set up immuneML for development`.

Step-by-step tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial, we provide a :code:`RandomDataPlot` (:download:`download here <./example_code/RandomDataPlot.py>` or view below), in order to test adding a new Report file to immuneML.
This report ignores the input dataset, and generates a scatterplot containing random values.

    .. collapse:: RandomDataPlot.py

      .. literalinclude:: ./example_code/RandomDataPlot.py
         :language: python

#. Add a new `Python package <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_ to the :py:mod:`~immuneML.reports.data_reports` package.
   This means: a new folder (with meaningful name) containing an empty :code:`__init__.py` file.

#. Add a new class to the :py:mod:`~immuneML.reports.data_reports` package (other reports types should be placed in the appropriate sub-package of :py:mod:`~immuneML.reports`).
   The new class should inherit from the base class :py:obj:`~immuneML.reports.data_reports.DataReport.DataReport`.

#. If the encoder has any default parameters, they should be added in a default parameters YAML file. This file should be added to the folder :code:`config/default_params/reports`.
   The default parameters file is automatically discovered based on the name of the class using the base name converted to snake case, and with an added '_params.yaml' suffix.
   For the :code:`RandomDataPlot`, this is :code:`random_data_report_params.yaml`, which could for example contain the following:

   .. code:: yaml

      n_points_to_plot: 10

   In rare cases where classes have unconventional names that do not translate well to CamelCase (e.g., MiXCR, VDJdb), this needs to be accounted for in :py:meth:`~immuneML.dsl.DefaultParamsLoader.convert_to_snake_case`.

Test running the new report with a YAML specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use immuneML directly to test run your report, the YAML example below may be used.
This example analysis creates a randomly generated dataset, and runs the :code:`RandomDataPlot` (which ignores the dataset).

   .. collapse:: test_run_random_data_report.yaml

      .. code:: yaml

         definitions:
           datasets:
             my_dataset:
               format: RandomSequenceDataset
               params:
                 sequence_count: 100

           reports:
             my_random_report: RandomDataPlot:
               n_points_to_plot: 10

         instructions:
           my_instruction:
             type: ExploratoryAnalysis
             analyses:
               my_analysis_1:
                 dataset: my_dataset
                 report: my_random_report



Adding a unit test for a Report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a unit test for the new :code:`RandomDataPlot` (:download:`download <./example_code/_test_randomDataPlot.py>` the example testfile or view below)

    .. collapse:: test_randomDataPlot.py

      .. literalinclude:: ./example_code/_test_randomDataPlot.py
         :language: python


#. Add a new file to :py:mod:`~test.reports.data_reports` package named test_randomDataPlot.py.
#. Add a class :code:`TestRandomDataPlot` that inherits :code:`unittest.TestCase` to the new file.
#. Add a function :code:`setUp()` to set up cache used for testing. This should ensure that the cache location will be set to :code:`EnvironmentSettings.tmp_test_path / "cache/"`
#. Define one or more tests for the class and functions you implemented.

   - It is recommended to at least test building and generating the report
   - Mock data is typically used to test new classes. Tip: the :code:`RandomDatasetGenerator` class can be used to generate Repertoire, Sequence or Receptor datasets with random sequences.
   - If you need to write data to a path (for example test datasets or results), use the following location: :code:`EnvironmentSettings.tmp_test_path / "some_unique_foldername"`


Implementing a new Report
----------------------------------------

This section describes tips and tricks for implementing your own new :code:`Report` from scratch.
Detailed instructions of how to implement each method, as well as some special cases, can be found in the
:py:obj:`~immuneML.ml_methods.classifiers.Report.Report` base class.

The :code:`Report` type is determined by subclassing one of the following:

#. :py:obj:`~immuneML.reports.data_reports.DataReport.DataReport` – reports examining some aspect of the dataset (such as sequence length distribution, gene usage)
#. :py:obj:`~immuneML.reports.encoding_reports.EncodingReport.EncodingReport` – shows some aspect of the encoded dataset (such as the feature values of an encoded dataset),
#. :py:obj:`~immuneML.reports.ml_reports.MLReport.MLReport` – shows the characteristics of an inferred machine learning model (such as coefficient values for logistic regression or kernel visualization for CNN)
#. :py:obj:`~immuneML.reports.train_ml_model_reports.TrainMLModelReport.TrainMLModelReport` – show statistics of multiple trained ML models in the TrainMLModelInstruction (such as comparing performance statistics between models, or performance w.r.t. an encoding parameter)
#. :py:obj:`~immuneML.reports.multi_dataset_reports.MultiDatasetReport.MultiDatasetReport` –  show statistics when running immuneML with the MultiDatasetBenchmarkTool


.. include:: ./coding_conventions_and_tips.rst


Determine the type of report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, it is important to determine what the type of the report is, as this defines which report class should be inherited.

Report types for dataset analysis
*************************************

If the report will be used to analyze a Dataset (such as a :code:`RepertoireDataset`), either a :code:`DataReport` or an :code:`EncodingReport` should be used. The simplest
report is the :code:`DataReport`, which should typically be used when summarizing some qualities of a dataset. This dataset can be found in the report
attribute dataset.

Use the :code:`EncodingReport` when it is necessary to access the encoded_data attribute of a :code:`Dataset`. The encoded_data attribute is an instance of a
:py:obj:`~immuneML.data_model.encoded_data.EncodedData.EncodedData` class. This report should be used when the data
representation first needs to be changed before running the report, either through an existing or a custom encoding (see:
:ref:`How to add a new encoding`). For example, the :ref:`Matches` report represents a RepertoireDataset based on matches to a given reference
dataset, and must first be encoded using a :ref:`MatchedSequences`, :ref:`MatchedReceptors` or :ref:`MatchedRegex`.

Report types for trained ML model analysis
********************************************

When the results of an experiment with a machine learning method should be analyzed, an :code:`MLReport` or :code:`TrainMLModelReport` should be used. These reports
are more advanced and require understanding of the :code:`TrainMLModelInstruction`. The :code:`MLReport` should be used when plotting statistics or
exporting information about one trained ML model. This report can be executed on any trained ML model (:code:`MLMethod` subclass object), both in the assessment and selection loop of
the :ref:`TrainMLModel`. An :code:`MLReport` has the following attributes:

  #. train_dataset: a Dataset (e.g., RepertoireDataset) object containing the training data used for the given classifier
  #. test_dataset: similar to train_dataset, but containing the test data
  #. method: the MLMethod object containing trained classifiers for each of the labels.
  #. label: the label that the report is executed for (the same report may be executed several times when training classifiers for multiple labels), can be used to retrieve relevant information from the MLMethod object.
  #. hp_setting: the :py:obj:`~immuneML.hyperparameter_optimization.HPSetting.HPSetting` object, containing all information about which preprocessing, encoding, and ML methods were used up to this point. This parameter can usually be ignored unless specific information from previous analysis steps is needed.

In contrast, :code:`TrainMLModelReport` is used to compare several [optimal] ML models. This report has access to the attribute state: a :py:obj:`~immuneML.hyperparameter_optimization.states.TrainMLModelState.TrainMLModelState`
object, containing information that has been collected through the execution of the :code:`TrainMLModelInstruction`. This includes all datasets, trained
models, labels, internal state objects for selection and assessment loops (nested cross-validation), optimal models, and more.

Finally, the :code:`MultiDatasetReport` is used in rare cases when running immuneML with the :code:`MultiDatasetBenchmarkTool`.
**This is an advanced report type and is not typically used.**
This report type can be used when comparing the performance of classifiers over several datasets and accumulating the results.
This report has the attribute instruction_states: a list of several :code:`TrainMLModelState` objects.

Input and output of the _generate() method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The abstract method `_generate()` must be implemented,
which has the following responsibilities:

  - It should create the report results, for example, compute the data or create the plots that should be returned by the report.
  - It should write the report results to the folder given at the variable :code:`self.result_path`.
  - It should return a :code:`ReportResult` object, which contains lists of :code:`ReportOutput` objects. These :code:`ReportOutput` objects simply contain the path to a figure, table, text, or another type of result.
    One report can have multiple outputs, as long as they are all referred to in the returned :code:`ReportResult` object. This is used to format the summary of the results in the HTML output file.
  - When the main result of the report is a plot, it is good practice to also make the raw data available to the user, for example as a csv file.


Creating plots
^^^^^^^^^^^^^^^^^^^^^^^^

The preferred method for plotting data is through `plotly <https://plotly.com/python/>`_, as it creates interactive and rescalable plots in HTML format [recommended] that
display nicely in the HTML output file. Alternatively, plots can also be in pdf, png, jpg and svg format.

.. note::

    When plotting data with `plotly <https://plotly.com/python/>`_, we recommend using the following color schemes for consistency:
    plotly.colors.sequential.Teal, plotly.colors.sequential.Viridis, or plotly.colors.diverging.Tealrose.
    Additionally, in the most of immuneML plots, 'plotly_white' theme is used for the background.

    For the overview of color schemes, visit `this link <https://plotly.com/python/builtin-colorscales/>`_.
    For plotly themes, visit `this link <https://plotly.com/python/templates/>`_.

Checking prerequisites and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New report objects are created by immuneML by calling the :code:`build_object()` method. This method can take in any custom parameters and should return an instance of the
report object. The parameters of the method :code:`build_object()` can be directly specified in the YAML specification, nested under the report type, for example:

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

Specifying different report types in YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Custom reports may be defined in the YAML specification under the key ‘definitions’ the same way as any other reports. The easiest way to test run
`Data reports <https://docs.immuneml.uio.no/specification.html#data-reports>`_ and `Encoding reports <https://docs.immuneml.uio.no/specification.html#encoding-reports>`_ is through the :ref:`ExploratoryAnalysis` instruction. They may also be specified in the :ref:`TrainMLModel`
instruction in the :code:`selection` and :code:`assessment` loop under :code:`reports:data_splits` and :code:`reports:encoding` respectively.

`ML model reports <https://docs.immuneml.uio.no/specification.html#ml-model-reports>`_ and `Train ML model reports <https://docs.immuneml.uio.no/specification.html#train-ml-model-reports>`_ can only be run through the :ref:`TrainMLModel` instruction. :ref:`ML reports` can be specified inside both the
:code:`selection` and :code:`assessment` loop under :code:`reports:models`. :ref:`Train ML model reports` must be specified under :code:`reports`.

Finally, :ref:`Multi dataset reports` can be specified under :code:`benchmark_reports` when running the :code:`MultiDatasetBenchmarkTool`.

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

Class documentation standards for reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ./class_documentation_standards.rst

.. collapse:: Click to view a full example of Report class documentation.

       .. code::

           This RandomDataPlot is a placeholder for a real Report.
           It plots some random numbers.

           **Specification arguments:**

           - n_points_to_plot (int): The number of random points to plot.


           **YAML specification:**

           .. indent with spaces
           .. code-block:: yaml

               my_report:
                   RandomDataPlot:
                       n_points_to_plot: 10

