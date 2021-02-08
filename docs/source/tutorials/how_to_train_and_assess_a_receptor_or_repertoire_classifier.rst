How to train and assess a receptor or repertoire-level ML classifier
====================================================================

immuneML provides a rich set of functionality for training and assessing machine
learning models to classify of receptors or repertoires. To learn the parameters and hyperparameters of the ML model,
the data needs to be split into training, validation and test sets. For this splitting, both nested cross-validation and fixed splits are supported.
Processing and filtering choices for input receptor data,
as well as encoding choice, can be set up as hyperparameters for automatic optimization
and unbiased assessment across such choices. One can set up a single immuneML run to train
models, optimize hyperparameters and get an unbiased assessment of its performance.
The resulting optimized classifier can also afterwards be applied to further datasets.
This process is shown in the figure below.

See :ref:`How to properly train and assess an ML model` to learn more about model training, hyperparameters and unbiased assessment.

.. figure:: ../_static/images/ml_process_overview.png
  :width: 70%

  Overview of the training process of an ML classifier: hyperparameter
  optimization is done on training and validation data and the model performance is
  assessed on test data

The analysis specification consists of (i) defining all elements used for analysis,
such as the dataset, encodings, preprocessing, ML methods and reports, (ii) defining
the instruction to be executed. Training ML model instructions take as parameters:

1. A list of hyperparameter :code:`settings` (:code:`encoding`, :code:`ml_method` and optional :code:`preprocessing` combinations) to be evaluated,

.. highlight:: yaml
.. code-block:: yaml
  :linenos:

  settings:
    - encoding: my_kmer_enc
      ml_method: my_log_reg
    - preprocessing: filter1
      encoding: my_kmer_enc
      ml_method: my_svm

2. :code:`assessment` configuration, including:

  2.1. What :code:`split_strategy` to use to split the data in the outer cross-validation loop,

  2.2. How many combinations of training/test datasets to generate based on the given splitting strategy (:code:`split_count`),

  2.3. What percentage of data to use for the training dataset (if splitting to training and test is random, :code:`training_percentage`),

  2.4. :code:`reports` to execute:

    2.4.1. :code:`models`: reports  to be generated for optimal models per label

    2.4.1. :code:`data`: reports to be executed on the whole dataset before it is split to training and test

    2.4.1. :code:`data_splits`: reports to be executed after the data has been split into training and test

    2.4.1. :code:`encoding`: reports to be executed on the encoded training and test datasets

  .. highlight:: yaml
  .. code-block:: yaml
    :linenos:

    assessment:
      split_strategy: random
      split_count: 5
      training_percentage: 0.7
      reports:
        models:
          - my_model_report
        data:
          - my_data_report
        data_splits:
          - my_data_report
        encoding:
          - my_encoding_report

3. :code:`selection` configuration, including:

  3.1. What :code:`split_strategy` to use to split the data in the inner cross-validation loop,

  3.2. How many combinations of training/test datasets to generate based on the given splitting strategy (:code:`split_count`),

  3.3. What percentage of data to use for the training dataset (if splitting to training and test is random, :code:`training_percentage`),

  3.4. :code:`reports` to execute:

    3.4.1. :code:`models`: reports to be executed on all trained classifiers

    3.4.2. :code:`data`: reports to be executed on the training dataset split before it is split to training and validation

    3.4.3. :code:`data_splits`: reports to be executed after the data has been split into training and validation

    3.4.4. :code:`encoding`: reports to be executed on the encoded training and validation datasets

  .. highlight:: yaml
  .. code-block:: yaml
    :linenos:

    selection:
      split_strategy: random
      split_count: 1
      reports:
        models:
          - my_model_report
        data:
          - my_data_report
        data_splits:
          - my_data_report
        encoding:
          - my_encoding_report
      training_percentage: 0.7

4. A list of :code:`labels` to use for prediction,

5. A list of :code:`metrics` for evaluation (e.g., :code:`accuracy`, :code:`balanced_accuracy`, :code:`f1_weighted`, ...),

6. A metric which will be used for evaluation (given under :code:`optimization_metric` field)

7. A list of :code:`reports` to be executed after the instruction has finished to show the overall performance

An example is shown below:

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      simulated_d1:
        format: AIRR
        params:
          metadata_file: path/to/metadata.csv
          path: path/to/data/
    encodings:
      my_kmer_enc:
        KmerFrequency:
          k: 4
          sequence_encoding: CONTINUOUS_KMER
          normalization_type: RELATIVE_FREQUENCY
      my_kmer_enc2:
        KmerFrequency:
          k: 3
          sequence_encoding: CONTINUOUS_KMER
          normalization_type: RELATIVE_FREQUENCY
    ml_methods:
      my_svm: SVM
      my_log_reg:
        LogisticRegression:
          penalty: l1
          C:
            - 1000
            - 100
            - 0.01
            - 0.001
        model_selection_cv: True
        model_selection_n_folds: 5
    reports:
      my_report: MLSettingsPerformance

  instructions:
    my_training_instruction:
      type: TrainMLModel
      settings:
        - encoding: my_kmer_enc
          ml_method: my_log_reg
        - encoding: my_kmer_enc2
          ml_method: my_svm
      assessment:
        split_strategy: random
        split_count: 5
        training_percentage: 0.7
      selection:
        split_strategy: random
        split_count: 1
        training_percentage: 0.7
      labels:
        - label
      dataset: simulated_d1
      metrics: [accuracy, auc] # metrics to be computed for all settings
      strategy: GridSearch
      number_of_processes: 4
      optimization_metric: balanced_accuracy # the metric used for optimization
      reports: [my_report]
      refit_optimal_model: False
      store_encoded_data: False


..
  left out, too detailed:

    The flow of the hyperparameter optimization is shown below, along with the
    output that is generated and reports executed at each step:

    .. figure:: ../_static/images/hp_optmization_with_outputs.png
      :width: 70%

      Execution flow of the TrainMLModelInstruction along with the information on data and reports generated at each step.

    For implementation detals, see :ref:`Hyperparameter Optimization Details`.