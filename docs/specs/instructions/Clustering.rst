



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


