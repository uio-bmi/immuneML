


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


