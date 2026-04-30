


Class implementing hyperparameter optimization and training and assessing the model through nested cross-validation (CV).
The process is defined by two loops:

- the outer loop over defined splits of the dataset for performance assessment

- the inner loop over defined hyperparameter space and with cross-validation or train & validation split
  to choose the best hyperparameters.

Optimal model chosen by the inner loop is then retrained on the whole training dataset in the outer loop.

.. note::

    If you are interested in plotting the performance of all combinations of encodings and ML methods on the test set,
    consider running the :ref:`MLSettingsPerformance` report as hyperparameter report in the assessment loop.


**Specification arguments:**

- dataset: dataset to use for training and assessing the classifier

- strategy: how to search different hyperparameters; common options include grid search, random search. Valid values are: `GridSearch`.

- settings (list): a list of combinations of `preprocessing_sequence`, `encoding` and `ml_method`. `preprocessing_sequence` is optional, while `encoding` and `ml_method` are mandatory. These three options (and their parameters) can be optimized over, choosing the highest performing combination.

- assessment: description of the outer loop (for assessment) of nested cross-validation. It describes how to split the data, how many splits to make, what percentage to use for training and what reports to execute on those splits. See plitConfig below.

- selection: description of the inner loop (for selection) of nested cross-validation. The same as assessment argument, just to be executed in the inner loop. See plitConfig below.

- metrics (list): a list of metrics (`accuracy`, `balanced_accuracy`, `confusion_matrix`, `f1_micro`, `f1_macro`, `f1_weighted`, `precision`, `precision_micro`, `precision_macro`, `precision_weighted`, `recall_micro`, `recall_macro`, `recall_weighted`, `average_precision`, `brier_score`, `recall`, `auc`, `auc_ovo`, `auc_ovr`, `log_loss`, `specificity`) to compute for all splits and settings created during the nested cross-validation. These metrics will be computed only for reporting purposes. For choosing the optimal setting, `optimization_metric` will be used.

- optimization_metric: a metric to use for optimization and assessment in the nested cross-validation (one of `accuracy`, `balanced_accuracy`, `confusion_matrix`, `f1_micro`, `f1_macro`, `f1_weighted`, `precision`, `precision_micro`, `precision_macro`, `precision_weighted`, `recall_micro`, `recall_macro`, `recall_weighted`, `average_precision`, `brier_score`, `recall`, `auc`, `auc_ovo`, `auc_ovr`, `log_loss`, `specificity`).

- example_weighting: which example weighting strategy to use. Example weighting can be used to up-weight or down-weight the importance of each example in the dataset. These weights will be applied when computing (optimization) metrics, and are used by some encoders and ML methods.

- labels (list): a list of labels for which to train the classifiers. The goal of the nested CV is to find the
  setting which will have best performance in predicting the given label (e.g., if a subject has experienced an immune event or not).
  Performance and optimal settings will be reported for each label separately. If a label is binary, instead of specifying only its name, one
  should explicitly set the name of the positive class as well under parameter `positive_class`. If positive class is not set, one of the label
  classes will be assumed to be positive.

- number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.

- reports (list): a list of report names to be executed after the nested CV has finished to show the overall performance or some statistic;
  the reports that can be provided here are :ref:`**Train ML model reports**`.

- refit_optimal_model (bool): if the final combination of preprocessing-encoding-ML model should be refitted on the full dataset thus providing
  the final model to be exported from instruction; alternatively, train combination from one of the assessment folds will be used

- export_all_models (bool): if set to True, all trained models in the assessment split are exported as .zip files.
  If False, only the optimal model is exported. By default, export_all_models is False.

- sequence_type (str): whether to perform the analysis on amino acid or nucleotide sequences

- region_type (str): which part of the sequence to analyze, e.g., IMGT_CDR3


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_nested_cv_instruction: # user-defined name of the instruction
            type: TrainMLModel # which instruction should be executed
            settings: # a list of combinations of preprocessing, encoding and ml_method to optimize over
                - preprocessing: seq1 # preprocessing is optional
                  encoding: e1 # mandatory field
                  ml_method: simpleLR # mandatory field
                - preprocessing: seq1 # the second combination
                  encoding: e2
                  ml_method: simpleLR
            assessment: # outer loop of nested CV
                split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
                split_count: 1 # how many train/test datasets to generate
                training_percentage: 0.7 # what percentage of the original data should be used for the training set
                reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
                    data_splits: # list of reports to execute on training/test datasets (before they are encoded)
                        - rep1
                    encoding: # list of reports to execute on encoded training/test datasets
                        - rep2
                    models: # list of reports to execute on trained ML methods for each assessment CV split
                        - rep3
            selection: # inner loop of nested CV
                split_strategy: k_fold # perform k-fold CV
                split_count: 5 # how many fold to create: here these two parameters mean: do 5-fold CV
                reports:
                    data_splits: # list of reports to execute on training/test datasets (in the inner loop, so these are actually training and validation datasets)
                        - rep1
                    models: # list of reports to execute on trained ML methods for each selection CV split
                        - rep2
                    encoding: # list of reports to execute on encoded training/test datasets (again, it is training/validation here)
                        - rep3
            labels: # list of labels to optimize the classifier for, as given in the metadata for the dataset
                - celiac:
                    positive_class: + # if it's binary classification, positive class parameter should be set
                - T1D # this is not binary label, so no need to specify positive class
            dataset: d1 # which dataset to use for the nested CV
            strategy: GridSearch # how to choose the combinations which to test from settings (GridSearch means test all)
            metrics: # list of metrics to compute for all settings, but these do not influence the choice of optimal model
                - accuracy
                - auc
            reports: # list of reports to execute when nested CV is finished to show overall performance
                - rep4
            number_of_processes: 4 # number of parallel processes to create (could speed up the computation)
            optimization_metric: balanced_accuracy # the metric to use for choosing the optimal model and during training
            refit_optimal_model: False # use trained model, do not refit on the full dataset
            export_all_ml_settings: False # only export the optimal setting
            region_type: IMGT_CDR3
            sequence_type: AMINO_ACID



**SplitConfig**

SplitConfig describes how to split the data for cross-validation. It allows for the following combinations:

- loocv (leave-one-out cross-validation)

- k_fold (k-fold cross-validation)

- stratified_k_fold (stratified k-fold cross-validation that can be used when immuneML is used for single-label
  classification, see `this documentation <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_ for more details on how this is implemented)

- random (Monte Carlo cross-validation - randomly splitting the dataset to training and test datasets)

- manual (train and test dataset are explicitly specified by providing metadata files for the two datasets)

- leave_one_out_stratification (leave-one-out CV where one refers to a specific parameter, e.g. if subject is known
  in a receptor dataset, it is possible to have leave-subject-out CV; or if a dataset contains multiple batches, it
  is possible to split evaluation by batch).

**Specification arguments:**

- split_strategy: one of the types of cross-validation listed above (`LOOCV`, `K_FOLD`, `STRATIFIED_K_FOLD`, `MANUAL`, ``  or `RANDOM`)

- split_count (int): if split_strategy is `K_FOLD`, then this defined how many splits to make (K), if split_strategy is RANDOM, split_count defines how many random splits to make, resulting in split_count training/test dataset pairs, or if split_strategy is `LOOCV`, `MANUAL` or `LEAVE_ONE_OUT_STRATIFICATION`, split_count does not need to be specified.

- training_percentage: if split_strategy is RANDOM, this defines which portion of the original dataset to use for creating the training dataset; for other values of split_strategy, this parameter is not used.

- reports: defines which reports to execute on which datasets or settings. See ReportConfig for more details.

- manual_config: if split strategy is `MANUAL`,
  here the paths to metadata files should be given (fields `train_metadata_path` and `test_metadata_path`). The matching of examples is done
  using the "subject_id" field in for repertoire datasets so it has to be present in both the original dataset and the metadata files provided
  here. For receptor and sequence datasets, "example_id" field needs to be provided in the metadata files and it will be mapped to either
  'sequence_identifiers' or 'receptor_identifiers' in the original dataset. If split strategy is anything other than `MANUAL`, this field has
  no effect and can be omitted.

- leave_one_out_config: if split strategy is
  `LEAVE_ONE_OUT_STRATIFICATION`, this config describes which parameter to use for stratification thus making a list of train/test dataset
  combinations in which in the test set there are examples with only one value of the specified parameter. `leave_one_out_config` argument
  accepts two inputs: `parameter` which is the name of the parameter to use for stratification and `min_count` which defines the minimum
  number of examples that can be present in the test dataset. This type of generating train and test datasets is only supported for receptor
  and sequence datasets so far. If split strategy is anything else, this field has no effect and can be omitted.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    # as a part of a TrainMLModel instruction, defining the outer (assessment) loop of nested cross-validation:
    assessment: # outer loop of nested CV
        split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
        split_count: 5 # how many train/test datasets to generate
        training_percentage: 0.7 # what percentage of the original data should be used for the training set
        reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
            data_splits: # list of data reports to execute on training/test datasets (before they are encoded)
                - rep1
            encoding: # list of encoding reports to execute on encoded training/test datasets
                - rep2
            models: # list of ML model reports to execute on the trained classifiers in the assessment loop
                - rep3

    # as a part of a TrainMLModel instruction, defining the inner (selection) loop of nested cross-validation:
    selection: # inner loop of nested CV
        split_strategy: leave_one_out_stratification
        leave_one_out_config: # perform leave-(subject)-out CV
            parameter: subject # which parameter to use for splitting, must be present in the metadata for each example
            min_count: 1 # what is the minimum number of examples with unique value of the parameter specified above for the analysis to be valid
        reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
            data_splits: # list of data reports to execute on training/test datasets (before they are encoded)
                - rep1
            encoding: # list of encoding reports to execute on encoded training/test datasets
                - rep2
            encoding: # list of ML model reports to execute the trained classifiers in the selection loop
                - rep3





**ReportConfig**

A class encapsulating different report lists which can be executed while performing nested cross-validation (CV) using TrainMLModel
instruction. All arguments are optional.

**Specification arguments:**

- data: :ref:`**Data reports**` to be executed on the whole dataset before it is split to training/test or training/validation

- data_splits: :ref:`**Data reports**` to be executed after the data has been split into training and test (assessment CV loop) or training and validation (selection CV loop) datasets before they are encoded

- models: :ref:`**ML model reports**` to be executed on all trained classifiers

- encoding: :ref:`**Encoding reports**` to be executed on each of the encoded training/test datasets or training/validation datasets

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    # as a part of a TrainMLModel instruction, defining the outer (assessment) loop of nested cross-validation:
    assessment: # outer loop of nested CV
        split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
        split_count: 5 # how many train/test datasets to generate
        training_percentage: 0.7 # what percentage of the original data should be used for the training set
        reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
            data_splits: # list of reports to execute on training/test datasets (before they are preprocessed and encoded)
                - my_data_split_report
            encoding: # list of reports to execute on encoded training/test datasets
                - my_encoding_report

    # as a part of a TrainMLModel instruction, defining the inner (selection) loop of nested cross-validation:
    selection: # inner loop of nested CV
        split_strategy: random # perform Monte Carlo CV (randomly split the data into train and validation)
        split_count: 5 # how many train/validation datasets to generate
        training_percentage: 0.7 # what percentage of the original data should be used for the training set
        reports: # reports to execute on training/validation datasets, encoded datasets and trained ML methods
            data_splits: # list of reports to execute on training/validation datasets (before they are preprocessed and encoded)
                - my_data_split_report
            encoding: # list of reports to execute on encoded training/validation datasets
                - my_encoding_report
            models:
                - my_ml_model_report




