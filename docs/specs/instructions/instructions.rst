ApplyGenModel
---------------------------





ApplyGenModel instruction implements applying generative AIRR models on the sequence level.

This instruction takes as input a trained model (trained in the :ref:`TrainGenModel` instruction)
which will be used for generating data and the number of sequences to be generated.
It can also produce reports of the applied model and reports of generated sequences.


**Specification arguments:**

- gen_examples_count (int): how many examples (sequences, repertoires) to generate from the applied model

- reports (list): list of report ids (defined under definitions/reports) to apply after generating
  gen_examples_count examples; these can be data reports (to be run on generated examples), ML reports (to be run
  on the fitted model)

- ml_config_path (str): path to the trained model in zip format (as provided by TrainGenModel instruction)

**YAML specification:**

.. highlight:: yaml
.. code-block:: yaml

    instructions:
        my_apply_gen_model_inst: # user-defined instruction name
            type: ApplyGenModel
            gen_examples_count: 100
            ml_config_path: ./config.zip
            reports: [data_rep1, ml_rep2]




Clustering
---------------------------





Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. It provides results
either for the full discovery dataset or for multiple subsets of discovery data as way to assess the stability
of different metrics (Liu et al., 2022; Dangl and Leisch, 2020; Lange et al. 2004). Finally, it
provides options to include a set of reports to visualize the results.

See also: :ref:`How to perform clustering analysis` for more details on the clustering procedure.

References:

Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-Based Validation of Clustering Solutions.
Neural Computation, 16(6), 1299–1323. https://doi.org/10.1162/089976604773717621

Dangl, R., & Leisch, F. (2020). Effects of Resampling in Determining the Number of Clusters in a Data Set.
Journal of Classification, 37(3), 558–583. https://doi.org/10.1007/s00357-019-09328-2

Liu, T., Yu, H., & Blair, R. H. (2022). Stability estimation for unsupervised clustering: A review. WIREs
Computational Statistics, 14(6), e1575. https://doi.org/10.1002/wics.1575

**Specification arguments:**

- dataset (str): name of the dataset to be clustered

- metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
  metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
  clusters can be compared against a list of predefined labels); some of the supported metrics include adjusted_rand_score,
  completeness_score, homogeneity_score, silhouette_score; for the full list, see scikit-learn's documentation of
  clustering metrics at https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster.

- labels (list): an optional list of labels to use for external evaluation of clustering

- sample_config (SampleConfig): configuration describing how to construct the data subsets to estimate different
  clustering settings' performance with different internal and external validation indices; with parameters
  `percentage`, `split_count`, `random_seed`:

.. indent with spaces
.. code-block:: yaml

    sample_config: # make 5 subsets with 80% of the data each
        split_count: 5
        percentage: 0.8
        random_seed: 42

- stability_config (StabilityConfig): configuration describing how to compute clustering stability;
  currently, clustering stability is computed following approach by Lange et al. (2004) and only takes the number
  of repetitions as a parameter. Other strategies to compute clustering stability will be added in the future.

.. indent with spaces
.. code-block:: yaml

    stability_config:
        split_count: 5 # number of times to repeat clustering for stability estimation
        random_seed: 12

- clustering_settings (list): a list where each element represents a :py:obj:`~immuneML.workflows.clustering.clustering_run_model.ClusteringSetting`; a combinations of encoding,
  optional dimensionality reduction algorithm, and the clustering algorithm that will be evaluated

- reports (list): a list of reports to be run on the clustering results or the encoded data

- number_of_processes (int): how many processes to use for parallelization

- sequence_type (str): whether to do analysis on the amino_acid or nucleotide level; this value is used only if
  nothing is specified on the encoder level

- region_type (str): which part of the receptor sequence to analyze (e.g., IMGT_CDR3); this value is used only if
  nothing is specified on the encoder level

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_clustering_instruction:
            type: Clustering
            dataset: d1
            metrics: [adjusted_rand_score, adjusted_mutual_info_score]
            labels: [epitope, v_call]
            sequence_type: amino_acid
            region_type: imgt_cdr3
            sample_config:
                split_count: 5
                percentage: 0.8
                random_seed: 42
            stability_config:
                split_count: 5
                random_seed: 12
            clustering_settings:
                - encoding: e1
                  dim_reduction: pca
                  method: k_means1
                - encoding: e2
                  method: dbscan
            reports: [rep1, rep2]




DatasetExport
---------------------------




DatasetExport instruction takes a list of datasets as input, optionally applies preprocessing steps, and outputs
the data in specified formats.

**Specification arguments:**

- datasets (list): a list of datasets to export in all given formats

- preprocessing_sequence (str): which preprocessing sequence to use on the dataset(s), this item is optional and does not have to be specified.
  When specified, the same preprocessing sequence will be applied to all datasets.

- number_of_processes (int): how many processes to use during repertoire export (not used for sequence datasets)

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_dataset_export_instruction: # user-defined instruction name
            type: DatasetExport # which instruction to execute
            datasets: # list of datasets to export
                - my_generated_dataset
                - my_dataset_from_adaptive
            preprocessing_sequence: my_preprocessing_sequence
            number_of_processes: 4




ExploratoryAnalysis
---------------------------




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



FeasibilitySummary
---------------------------




FeasibilitySummary instruction creates a small synthetic dataset and reports summary metrics to show if the simulation with the given
parameters is feasible. The input parameters to this analysis are the name of the simulation
(the same that can be used with LigoSim instruction later if feasibility analysis looks acceptable), and the number of sequences to
simulate for estimating the feasibility.

The feasibility analysis is performed for each generative model separately as these could differ in the analyses that will be reported.

**Specification arguments:**

- simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how to combine signals with simulated data; specified under definitions

- sequence_count (int): how many sequences to generate to estimate feasibility (default value: 100 000)

- number_of_processes (int): for the parts of the analysis that are possible to parallelize, how many processes to use


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_feasibility_summary: # user-defined name of the instruction
            type: FeasibilitySummary # which instruction to execute
            simulation: sim1
            sequence_count: 10000




LigoSim
---------------------------




LIgO simulation instruction creates a synthetic dataset from scratch based on the generative model and a set of signals provided by
the user.

**Specification arguments:**

- simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how to combine signals with simulated data; specified under definitions

- sequence_batch_size (int): how many sequences to generate at once using the generative model before checking for signals and filtering

- max_iterations (int): how many iterations are allowed when creating sequences

- export_p_gens (bool): whether to compute generation probabilities (if supported by the generative model) for sequences and include them as part of output

- number_of_processes (int): determines how many simulation items can be simulated in parallel


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_simulation_instruction: # user-defined name of the instruction
            type: LIgOSim # which instruction to execute
            simulation: sim1
            sequence_batch_size: 1000
            max_iterations: 1000
            export_p_gens: False
            number_of_processes: 4




MLApplication
---------------------------




Instruction which enables using trained ML models and encoders on new datasets which do not necessarily have labeled data.
When the same label is provided as the ML setting was trained for, performance metrics can be computed.

The predictions are stored in the predictions.csv in the result path in the following format:

.. list-table::
    :widths: 25 25 25 25
    :header-rows: 1

    * - example_id
      - cmv_predicted_class
      - cmv_1_proba
      - cmv_0_proba
    * - e1
      - 1
      - 0.8
      - 0.2
    * - e2
      - 0
      - 0.2
      - 0.8
    * - e3
      - 1
      - 0.78
      - 0.22


If the same label that the ML setting was trained for is present in the provided dataset, the 'true' label value
will be added to the predictions table in addition:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - example_id
      - cmv_predicted_class
      - cmv_1_proba
      - cmv_0_proba
      - cmv_true_class
    * - e1
      - 1
      - 0.8
      - 0.2
      - 1
    * - e2
      - 0
      - 0.2
      - 0.8
      - 0
    * - e3
      - 1
      - 0.78
      - 0.22
      - 0

**Specification arguments:**

- dataset: dataset for which examples need to be classified

- config_path: path to the zip file exported from MLModelTraining instruction (which includes train ML model, encoder, preprocessing etc.)

- number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.

- metrics (list): a list of metrics (`accuracy`, `balanced_accuracy`, `confusion_matrix`, `f1_micro`, `f1_macro`, `f1_weighted`, `precision`, `precision_micro`, `precision_macro`, `precision_weighted`, `recall_micro`, `recall_macro`, `recall_weighted`, `average_precision`, `brier_score`, `recall`, `auc`, `auc_ovo`, `auc_ovr`, `log_loss`, `specificity`) to compute between the true and predicted classes. These metrics will only be computed when the same label with the same classes is provided for the dataset as the original label the ML setting was trained for.


**YAML specification:**

.. highlight:: yaml
.. code-block:: yaml

    instructions:
        instruction_name:
            type: MLApplication
            dataset: d1
            config_path: ./config.zip
            metrics:
            - accuracy
            - precision
            - recall
            number_of_processes: 4




SplitDataset
---------------------------




This instruction splits the dataset into two as defined by the instruction parameters. It can be used as a first
step in clustering to obtain discovery and validation datasets, or to leave out the test dataset for classification.

For classification, :ref:`TrainMLModel` instruction can be used for more complex data splitting (e.g.,
nested cross-validation with different data splitting strategies).

**Specification arguments:**

- dataset (str): name of the dataset to split, as defined previously in the analysis specification

- split_config (SplitConfig): the split configuration; split_count has to be 1


**YAML specification:**

.. code-block:: yaml

    instructions:
        split_dataset1:
            type: SplitDataset
            dataset: d1
            split_config:
                split_count: 1
                split_strategy: random
                training_percentage: 0.5




Subsampling
---------------------------




Subsampling is an instruction that subsamples a given dataset and creates multiple smaller dataset according to the
parameters provided.

**Specification arguments:**

- dataset (str): original dataset which will be used as a basis for subsampling

- subsampled_dataset_sizes (list): a list of dataset sizes (number of examples) each subsampled dataset should have

- subsampled_repertoire_size (int): the number of sequences to keep per repertoire (or None if all sequences should
  be kept) if dataset is a RepertoireDataset; otherwise, this argument is ignored.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_subsampling_instruction: # user-defined name of the instruction
            type: Subsampling # which instruction to execute
            dataset: my_dataset # original dataset to be subsampled, with e.g., 300 examples
            subsampled_dataset_sizes: # how large the subsampled datasets should be, one dataset will be created for each list item
                - 200 # one subsampled dataset with 200 examples (200 repertoires if my_dataset was repertoire dataset)
                - 100 # the other subsampled dataset will have 100 examples




TrainGenModel
---------------------------





TrainGenModel instruction implements training generative AIRR models on receptor level. Models that can be trained
for sequence generation are listed under Generative Models section.

This instruction takes a dataset as input which will be used to train a model, the model itself, and the number of
sequences to generate to illustrate the applicability of the model. It can also produce reports of the fitted model
and reports of original and generated sequences.

To use the generative model previously trained with immuneML, see :ref:`ApplyGenModel` instruction.


**Specification arguments:**

- dataset: dataset to use for fitting the generative model; it has to be defined under definitions/datasets

- methods: which methods to fit (defined previously under definitions/ml_methods); for compatibility with previous
  versions 'method' with a single method can also be used, but the single method option will be removed in the
  future.

- number_of_processes (int): how many processes to use for fitting the model

- gen_examples_count (int): how many examples (sequences, repertoires) to generate from the fitted model

- reports (list): list of report ids (defined under definitions/reports) to apply after fitting a generative model
  and generating gen_examples_count examples; these can be data reports (to be run on generated examples), ML
  reports (to be run on the fitted model)

- split_strategy (str): strategy to use for splitting the dataset into training and test datasets; valid options are
  RANDOM and MANUAL (in which case train and test set are fixed); default is RANDOM

- training_percentage (float): percentage of the dataset to use for training the generative model if split_strategy
  parameter is RANDOM. If set to 1, the
  full dataset will be used for training and the test dataset will be the same as the training dataset. Default
  value is 0.7. When export_combined_dataset is set to True, the splitting of sequences into train, test, and
  generated will be shown in column dataset_split.

- manual_config (dict): if split_strategy is set to MANUAL, this parameter can be used to specify the ids of examples
  that should be in train and test sets; the paths to csv files with ids for train and test data should be provided
  under keys 'train_metadata_path' and 'test_metadata_path'

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_train_gen_model_inst: # user-defined instruction name
            type: TrainGenModel
            dataset: d1 # defined previously under definitions/datasets
            methods: [model1] # defined previously under definitions/ml_methods
            gen_examples_count: 100
            number_of_processes: 4
            training_percentage: 0.7
            split_strategy: RANDOM # optional, default is RANDOM
            export_generated_dataset: True
            export_combined_dataset: False
            reports: [data_rep1, ml_rep2]

        my_train_gen_model_with_manual_split: # another instruction example
            type: TrainGenModel
            dataset: d1 # defined previously under definitions/datasets
            methods: [m1, m2]
            gen_examples_count: 100
            split_strategy: MANUAL
            split_config:
                train_metadata_path: path/to/train_metadata.csv # path to csv file with ids of examples in train set
                test_metadata_path: path/to/test_metadata.csv # path to csv file with ids of examples in test set
            export_generated_dataset: True
            export_combined_dataset: False
            reports: [data_rep1, ml_rep2]




TrainMLModel
---------------------------




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






ValidateClustering
---------------------------




ValidateClustering instruction supports the application of the chosen clustering setting (preprocessing, encoding,
clustering, with all hyperparameters) to a new dataset for validation.

For more details on validating the clustering algorithm and its hyperparameters, see the paper:
Ullmann, T., Hennig, C., & Boulesteix, A.-L. (2022). Validation of cluster analysis results on validation
data: A systematic framework. WIREs Data Mining and Knowledge Discovery, 12(3), e1444.
https://doi.org/10.1002/widm.1444

**Specification arguments:**

- clustering_config_path (str): path to the clustering exported by the Clustering instruction that will be applied
  to the new dataset

- dataset (str): name of the validation dataset to which the clustering will be applied, as defined under definitions

- metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
  metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
  clusters can be compared against a list of predefined labels); some of the supported metrics include adjusted_rand_score,
  completeness_score, homogeneity_score, silhouette_score; for the full list, see scikit-learn's documentation of
  clustering metrics at https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster.

- validation_type (list): how to perform validation; options are `method_based` validation (refit the clustering
  algorithm to the new dataset and compare the clusterings) and `result_based` validation (transfer the clustering
  from the original dataset to the validation dataset using a supervised classifier and compare the clusterings)

- reports (list): a list of reports to run on the validation results; supported report types include:

  - ClusteringMethodReport: reports that analyze the clustering method results (e.g., ClusteringVisualization)
  - EncodingReport: reports that analyze the encoded dataset
  - DataReport: reports that analyze the raw dataset


**YAML specification:**

.. code-block:: yaml

    instructions:
        validate_clustering_inst:
            type: ValidateClustering
            clustering_config_path: /path/to/exported_clustering.zip
            dataset: val_dataset
            metrics: [adjusted_rand_score, silhouette_score]
            validation_type: [method_based, result_based]
            reports: [cluster_vis, encoding_report]




