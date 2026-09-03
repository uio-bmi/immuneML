


Allows exploratory analysis of different datasets using encodings and reports.

Analysis is defined by a dictionary of ExploratoryAnalysisUnit objects that encapsulate a dataset, an encoding [optional]
and a report to be executed on the [encoded] dataset. Each analysis specified under `analyses` is completely independent from all
others.

.. note::

    The "report" parameter has been updated to support multiple "reports" per analysis unit. For backward
    compatibility, the "report" key is still accepted, but it will be ignored if "reports" is present.
    "report" option will be removed in the next major version.

**Specification arguments:**

- analyses (dict): a dictionary of analyses to perform. The keys are the names of different analyses, and the values for each
  of the analyses are:

  - dataset: dataset on which to perform the exploratory analysis

  - preprocessing_sequence: which preprocessings to use on the dataset, this item is optional and does not have to be specified.

  - example_weighting: which example weighting strategy to use before encoding the data, this item is optional and does not have to be specified.

  - encoding: how to encode the dataset before running the report, this item is optional and does not have to be specified.

  - labels: if encoding is specified, the relevant labels should be specified here.

  - dim_reduction: which dimensionality reduction to apply;

  - reports: which reports to run on the dataset. Reports specified here may be of the category :ref:`**Data reports**`
    or :ref:`**Encoding reports**`, depending on whether 'encoding' was specified.

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
                    reports: [r1] # which reports to generate using the dataset d1
                my_second_analysis: # user-defined name of another analysis
                    dataset: d1 # dataset to use in the second analysis - can be the same or different from other analyses
                    encoding: e1 # encoding to apply on the specified dataset (d1)
                    reports: [r2] # which reports to generate in the second analysis
                    labels: # labels present in the dataset d1 which will be included in the encoded data on which report r2 will be run
                        - celiac # name of the first label as present in the column of dataset's metadata file
                        - CMV # name of the second label as present in the column of dataset's metadata file
                my_third_analysis: # user-defined name of another analysis
                    dataset: d1 # dataset to use in the second analysis - can be the same or different from other analyses
                    encoding: e1 # encoding to apply on the specified dataset (d1)
                    dim_reduction: umap # or None; which dimensionality reduction method to apply to encoded d1
                    reports: [r3] # which report to generate in the third analysis
            number_of_processes: 4 # number of parallel processes to create (could speed up the computation)

