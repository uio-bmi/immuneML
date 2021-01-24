How to add a new report
========================

In immuneML, it is possible to automatically generate a report describing some aspect of the problem being examined. There are a few types of reports:

  #. Data report – reports examining some aspect of the dataset (such as sequence length distribution, gene usage)
  #. Encoding report – shows some aspect of the encoded dataset (such as the feature values of an encoded dataset),
  #. ML model report – shows the characteristics of an inferred machine learning model (such as coefficient values for logistic regression or kernel visualization for CNN)
  #. Train ML model report – show statistics of multiple trained ML models in the TrainMLModelInstruction (such as comparing performance statistics between models, or performance w.r.t. an encoding parameter)
  #. Multi dataset report –  show statistics when running immuneML with the MultiDatasetBenchmarkTool

These types of reports are modeled by the following classes:

  #. :py:obj:`source.reports.data_reports.DataReport.DataReport`
  #. :py:obj:`source.reports.encoding_reports.EncodingReport.EncodingReport`
  #. :py:obj:`source.reports.ml_reports.MLReport.MLReport`
  #. :py:obj:`source.reports.train_ml_model_reports.TrainMLModelReport.TrainMLModelReport`
  #. :py:obj:`source.reports.multi_dataset_reports.MultiDatasetReport.MultiDatasetReport`

The existing reports can be found in the package `source.reports`. These can be specified in the YAML by specifying the name and optional parameters
(see: :ref:`How to specify an analysis with YAML`).

This guide describes how to add a new custom report to immuneML.

Creating a custom report
-------------------------

Determine the type of report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, it is important to determine what the type of the report is, as this defines which report class should be inherited.

If the report will be used to analyze a Dataset (such as a RepertoireDataset), either a DataReport or an EncodingReport should be used. The simplest
report is the DataReport, which should typically be used when summarizing some qualities of a Dataset. This Dataset can be found in the report
attribute dataset.

Use the EncodingReport when it is necessary to access the encoded_data attribute of a Dataset. This report should be used when the data
representation first needs to be changed before running the report, either through an existing or a custom encoding (see:
:ref:`How to add a new encoding`). For example, the :ref:`Matches` report represents a RepertoireDataset based on matches to a given reference
dataset, and must first be encoded using a MatchedSequencesEncoder, MatchedReceptorsEncoder or MatchedRegexEncoder.

When the results of an experiment with a machine learning method should be analyzed, an MLReport or TrainMLModelReport should be used. These reports
are a bit more advanced and require more understanding of the TrainMLModelInstruction. The MLReport should be used when plotting statistics or
exporting information about one trained ML model. This report can be executed on any trained ML model, both in the assessment and selection loop of
the :ref:`TrainMLModel`. An MLReport has the following attributes:

  #. train_dataset: a Dataset (e.g., RepertoireDataset) object containing the training data used for the given classifier
  #. test_dataset: similar to train_dataset, but containing the test data
  #. method: the MLMethod object containing trained classifiers for each of the labels.
  #. label: the label that the report is executed for (the same report may be executed several times when training classifiers for multiple labels), can be used to retrieve relevant information from the MLMethod object.
  #. hp_setting: the :py:obj:`source.hyperparameter_optimization.HPSetting.HPSetting` object, containing all information about which preprocessing, encoding, and ML methods were used up to this point. This parameter can usually be ignored unless specific information from previous analysis steps is needed.

In contrast, `TrainMLModelReport` is used to compare several [optimal] ML models. This report has access to the attribute state: a :py:obj:`source.hyperparameter_optimization.states.TrainMLModelState.TrainMLModelState`
object, containing information that has been collected through the execution of the TrainMLModelInstruction. This includes all datasets, trained
models, labels, internal state objects for selection and assessment loops (nested cross-validation), optimal models, and more.

Finally, the MultiDatasetReport is used in rare cases when running immuneML with the MultiDatasetBenchmarkTool. This can be used when comparing the
performance of classifiers over several datasets and accumulating the results. This report has the attribute instruction_states: a list of several
TrainMLModelState objects.

Implementing the report
^^^^^^^^^^^^^^^^^^^^^^^^

The new report should inherit the appropriate report type and be placed in the respective package (under `source.reports`, choose `data_reports`,
`encoding_reports`, `ml_reports`, `train_ml_model_reports`, or `multidataset_reports`). The abstract method `generate()` must be implemented,
which has the following responsibilities:

  - It should create the report results, for example, compute the data or create the plots that should be returned by the report.
  - It should write the report results to the folder given at the variable result_path.
  - It should return a ReportResult object, which contains lists of ReportOutput objects. These ReportOutput objects simply contain the path to a figure, table, text, or another type of result. One report can have multiple outputs, as long as they are all accessible through the ReportResult. This will be later used to format the summary of the results in the HTML output file.
  - When the main result of the report is a plot, it is good practice to also make the raw data available to the user, for example as a csv file.

The preferred method for plotting data is through `plotly <https://plotly.com/python/>`_, as it creates interactive and rescalable plots in HTML format [recommended] that
display nicely in the HTML output file. Alternatively, plots can also be in pdf, png, jpg and svg format.

The second abstract method to be implemented is `build_object()`. This method can take in any custom parameters and should return an instance of the
report object. The parameters of the method `build_object()` can be directly specified in the YAML specification, nested under the report type, for example:

.. code-block:: yaml

  MyNewReport:
    custom_parameter: “value”


Inside the `build_object()` method, you can check if the correct parameters are specified and raise an exception when the user input is incorrect
(for example using the :py:obj:`source.util.ParameterValidator.ParameterValidator` utility class). Furthermore, it is possible to resolve more
complex input parameters, such as loading reference sequences from an external input file, before passing them to the `__init__()` method of the report.

It is important to consider whether the method `check_prerequisites()` should be implemented. This method should return a boolean value describing
whether the prerequisites are met, and print a warning message to the user when this condition is false. The report will only be generated when
`check_prerequisites()` returns true. This method should not be used to raise exceptions. Instead, it is used to prevent exceptions from happening
during execution, as this might cause lost results. Situations to consider are:

  - When implementing an EncodingReport, use this function to check that the data has been encoded and that the correct encoder has been used.
  - Similarly, when creating an MLReport or TrainMLModelReport, check that the appropriate ML methods have been used.

.. include:: ./dev_docs_util.rst

.. note::

  Please see the :py:obj:`source.reports.Report.Report` class for the detailed description of the methods to be implemented.

Unit testing the new report
----------------------------

For each report, a unit test should be added under the correct package inside test.reports. Here, the `generate()` method of the new report should be
tested, as well as other relevant methods, to ensure that the report output is correct. When building tests for reports, a useful class is
:py:obj:`source.simulation.dataset_generation.RandomDatasetGenerator.RandomDatasetGenerator`, which can create a dataset with random sequences.

Adding documentation for the new report
-----------------------------------------

After implementing the desired functionality, the documentation for the report should be added, so that the users of immuneML have sufficient information when deciding to use the report. It should be added to the docstring and consist of the following components:

  #. A short description of what the report is meant for.
  #. Optional extended description, including any references or specific cases that should bee considered.
  #. List of arguments the report takes as input. If the report does not take any arguments other than the ones provided by the immuneML in runtime depending on the report type (such as training and test dataset or trained method), there should be only a short statement that the report does not take input arguments.
  #. An example of how the report can be specified in YAML.

Here is an example of a documentation for the :ref:`DesignMatrixExporter` report that has no input arguments which can be provided by the user in the YAML
specification (the encoded dataset to be exported will be provided by immuneML at runtime):

.. code-block:: text

  Exports the design matrix and related information of a given encoded Dataset to csv files. If the encoded data has more than 2 dimensions
  (such as when using the OneHot encoder with option Flatten=False), the data are instead exported to .npy format and can be imported later outside of
  immuneML using numpy package and numpy.load() function.

  There are no input arguments for this report.

  YAML specification:

  .. code-block:: yaml

      my_dme_report: DesignMatrixExporter

Here is an example of documentation for the :ref:`MLSettingsPerformance` report with user-defined input arguments:

.. code-block:: text

  Report for TrainMLModel instruction that plots the performance for each of the setting combinations as defined under 'settings' in the
  assessment (outer validation) loop.

  The performances are grouped by label (horizontal panels) encoding (vertical panels) and ML method (bar color).
  When multiple data splits are used, the average performance over the data splits is shown with an error bar
  representing the standard deviation.

  This report can be used only with TrainMLModel instruction under 'reports'.


  Arguments:

      single_axis_labels (bool): whether to use single axis labels. Note that using single axis labels makes the
      figure unsuited for rescaling, as the label position is given in a fixed distance from the axis. By default,
      single_axis_labels is False, resulting in standard plotly axis labels.

      x_label_position (float): if single_axis_labels is True, this should be an integer specifying the x axis label
      position relative to the x axis. The default value for label_position is -0.1.

      y_label_position (float): same as x_label_position, but for the y axis.

  YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_hp_report: MLSettingsPerformance


When the docstrings are defined in this way, the documentation generated from them can be used directly on the documentation website.

Test run of the report: specifying the new report in YAML
-----------------------------------------------------------

Custom reports may be defined in the YAML specification under the key ‘definitions’ the same way as any other reports. The easiest way to test run
`DataReports` and `EncodingReports` is through the `ExploratoryAnalysis` instruction. They may also be specified in the `TrainMLModelInstruction`
instruction in the ‘selection’ and ‘assessment’ loop under ‘reports:data_splits’ and ‘reports:encoding’ respectively.

`MLReports` and `TrainMLModelReports` can only be run through the `TrainMLModelInstruction` instruction. `MLReports` can be specified inside both the
‘selection’ and ‘assessment’ loop under ‘reports/models’. `TrainMLModelReports` must be specified under ‘reports’.

Finally, `MultiDatasetReports` multi dataset reports can be specified under 'benchmark_reports’ when running the `MultiDatasetBenchmarkTool`.

The following specification shows the places where `DataReports`, `EncodingReports`, `MLReports`, and `TrainMLModelReports` can be specified:

.. code-block:: yaml

  definitions:
    reports:
      my_data_report: MyNewDataReport # example data report without parameters
      my_encoding_report: # example encoding report with a parameter
        MyNewEncodingReport:
         parameter: value
      my_ml_report: MyNewMLReport # ml model report
      my_trainml_report: MyNewMLModelReport # ml report

    datasets:
      d1:
          ... # if you do not have real data to test your report with, consider
              # using a randomly generated dataset, see the documentation:
              # “How to generate a random receptor or repertoire dataset”
    encodings:
      e1:
          ...
    ml_methods:
      m1:
        ...

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
          Labels: # when running an encoding report, labels must be specified
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
        ...
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
        ...
      reports:
        - my_trainml_report
      labels:
        - disease
      ...
