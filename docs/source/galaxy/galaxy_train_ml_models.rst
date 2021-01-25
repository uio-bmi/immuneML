How to train ML models in Galaxy
=========================================

The Galaxy tool 'Train machine learning models' can be used to run hyperparameter optimization over several different ML settings,
which include ML models and their parameters, encodings and preprocessing steps. Nested cross-validation is used to identify the optimal combination of ML settings.

This is a YAML-based Galaxy tool, if you prefer a button-based interface that assumes less ML knowledge, see the tutorials for training ML models for
:ref:`receptor <How to train immune receptor classifiers using the easy Galaxy interface>` and :ref:`repertoire <How to train immune repertoire classifiers using the easy Galaxy interface>`
classification using the easy Galaxy interfaces.



Creating the YAML specification
---------------------------------------------
This Galaxy tool takes as input an immuneML dataset from the Galaxy history, optional additional files, and a YAML specification file.

To train ML models in immuneML, the :ref:`TrainMLModel` instruction should be used. One or more :ref:`ML methods` and :ref:`Encodings` must be used,
and in addition it is possible to include :ref:`Preprocessings` in the hyperparameter optimization. :ref:`Reports` may be specified to export
plots and statistics in order to gain more insight into the dataset or the process of training ML models.
Constructing a YAML for training ML models is described in more detail in the tutorial :ref:`How to train and assess a receptor/repertoire-level ML classifier`.

When writing an analysis specification for Galaxy, it can be assumed that all selected files are present in the current working directory. A path
to an additional file thus consists only of the filename. Note that in Galaxy, it is only possible to train ML models for one label at a time.

A complete YAML specification for training ML models is shown here:


.. highlight:: yaml
.. code-block:: yaml

    definitions:
      datasets:
        dataset: # user-defined dataset name
          format: Pickle # the default format used by the 'Create dataset' galaxy tool is Pickle
          params:
            path: dataset.iml_dataset # specify the dataset name, the default name used by
                                      # the 'Create dataset' galaxy tool is dataset.iml_dataset

      encodings:
        my_3mer_encoding: # user-defined encoding name
          KmerFrequency:
            k: 3
        my_5mer_encoding:
          KmerFrequency:
            k: 5

      ml_methods:
        my_logistic_regression:
          LogisticRegression:
            C:
            - 0.01
            - 0.1
            - 1
            - 10
            - 100
            show_warnings: false # disabling scikit-learn warnings is recommended for Galaxy users
          model_selection_cv: true     # use scikit-learns 5-fold cross-validation to search
          model_selection_n_folds: 5   # over the optimal values for hyperparameter C

      reports:
        my_benchmark: MLSettingsPerformance
        my_coefficients:
          Coefficients:
            coefs_to_plot:
            - N_LARGEST
            n_largest:
            - 25

    instructions:
      my_training_instruction: # user-defined instruction name
        type: TrainMLModel

        dataset: dataset # select the dataset defined above
        labels:          # only one label can be specified here
        - disease

        settings:        # which combinations of ML settings to run
        - encoding: my_3mer_encoding
          ml_method: my_logistic_regression
        - encoding: my_5mer_encoding
          ml_method: my_logistic_regression

        assessment: # parameters in the assessment (outer) cross-validation loop
          reports:
            models:
            - my_coefficients  # run the 'coefficients' report on all the models
          split_count: 3
          split_strategy: random
          training_percentage: 0.7
        selection:  # parameters in the assessment (inner) cross-validation loop
          split_count: 1
          split_strategy: random
          training_percentage: 0.7

        reports: # train ML model reports to run
        - my_benchmark

        strategy: GridSearch
        optimization_metric: balanced_accuracy
        metrics:
        - accuracy
        - balanced_accuracy
        number_of_processes: 10
        refit_optimal_model: true
        store_encoded_data: false


Tool output
---------------------------------------------
This Galaxy tool will produce the following history elements:

- ML Model Training Archive: a .zip file containing the complete output folder as it was produced by immuneML. This folder
  contains the output of the TrainMLModel instruction including all trained models and their predictions, and report results.
  Furthermore, the folder contains the complete YAML specification file for the immuneML run, the HTML output and a log file.

- Results of ML model training: a HTML page that allows you to browse through all results, including prediction accuracies on
  the various data splits and report results.

- Optimal ML model: a .zip file containing the raw files for the optimal trained ML model file for the given label.
  This .zip file can subsequently be used as an input when :ref:`applying previously trained ML models to a new AIRR dataset in Galaxy <How to apply previously trained ML models to a new AIRR dataset in Galaxy>`