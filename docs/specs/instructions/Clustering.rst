



Clustering instruction fits clustering methods to the provided encoded dataset and compares the combinations of
clustering method with its hyperparameters, and encodings across a pre-defined set of metrics. The dataset is split
into discovery and validation datasets and the clustering results are reported on both. Finally, it
provides options to include a set of reports to visualize the results.

See also: :ref:`How to perform clustering analysis`

For more details on choosing the clustering algorithm and its hyperparameters, see the paper:
Ullmann, T., Hennig, C., & Boulesteix, A.-L. (2022). Validation of cluster analysis results on validation
data: A systematic framework. WIREs Data Mining and Knowledge Discovery, 12(3), e1444.
https://doi.org/10.1002/widm.1444


**Specification arguments:**

- dataset (str): name of the dataset to be clustered

- metrics (list): a list of metrics to use for comparison of clustering algorithms and encodings (it can include
  metrics for either internal evaluation if no labels are provided or metrics for external evaluation so that the
  clusters can be compared against a list of predefined labels); some of the supported metrics include adjusted_rand_score,
  completeness_score, homogeneity_score, silhouette_score; for the full list, see scikit-learn's documentation of
  clustering metrics at https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster.

- labels (list): an optional list of labels to use for external evaluation of clustering

- split_config (SplitConfig): how to perform splitting of the original dataset into discovery and validation data;
  for this parameter, specify: split_strategy (leave_one_out_stratification, manual, random), training percentage
  if split_strategy is random, and defaults of manual or leave one out stratification config for corresponding split
  strategy; all three options are illustrated here:

  .. indent with spaces
  .. code-block:: yaml

    split_config:
        split_strategy: manual
        manual_config:
            discovery_data: file_with_ids_of_examples_for_discovery_data.csv
            validation_data: file_with_ids_of_examples_for_validation_data.csv

  .. indent with spaces
  .. code-block:: yaml

    split_config:
        split_strategy: random
        training_percentage: 0.5
        split_count: 3 # repeat the random split 3 times -> 3 discovery and 3 validation datasets

  .. indent with spaces
  .. code-block:: yaml

    split_config:
        split_strategy: leave_one_out_stratification
        leave_one_out_config:
            parameter: subject_id # any name of the parameter for split, must be present in the metadata
            min_count: 1 #  defines the minimum number of examples that can be present in the validation dataset.

- clustering_settings (list): a list where each element represents a :py:obj:`~immuneML.workflows.clustering.clustering_run_model.ClusteringSetting`; a combinations of encoding,
  optional dimensionality reduction algorithm, and the clustering algorithm that will be evaluated

- reports (list): a list of reports to be run on the clustering results or the encoded data

- number_of_processes (int): how many processes to use for parallelization

- sequence_type (str): whether to do analysis on the amino_acid or nucleotide level; this value is used only if
  nothing is specified on the encoder level

- region_type (str): which part of the receptor sequence to analyze (e.g., IMGT_CDR3); this value is used only if
  nothing is specified on the encoder level

- validation_type (list): a list of validation types to use for comparison of clustering algorithms and encodings;
  it can be method_based and/or result_based

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
            validation_type: [method_based, result_based]
            split_config:
                split_count: 1
                split_strategy: manual
                manual_config:
                    discovery_data: file_with_ids_of_examples_for_discovery_data.csv
                    validation_data: file_with_ids_of_examples_for_validation_data.csv
            clustering_settings:
                - encoding: e1
                  dim_reduction: pca
                  method: k_means1
                - encoding: e2
                  method: dbscan
            reports: [rep1, rep2]


