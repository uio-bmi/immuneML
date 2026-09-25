How to perform clustering analysis
=====================================================================

In this tutorial, we will perform clustering analysis on a dataset of immune receptor sequences.
The dataset was simulated to contain sequences sharing strong structural resemblance — each sequence carries one
of 15 distinct implanted motifs.

The dataset (:download:`simulated_highly_similar_sequences.tsv <simulated_highly_similar_sequences.tsv>`) contains 14,724 TRB sequences in AIRR format. Each sequence
is annotated with a ``motif`` label indicating which of the 15 motifs was implanted, but this label is only used
for external validation — not as input to clustering.

The analysis proceeds in four steps:

1. Split the dataset into a discovery set and a validation set.
2. Run clustering analysis on the discovery set to explore different combinations of encoding and clustering algorithm.
3. Manually inspect the results and choose the best-performing clustering setting.
4. Validate the chosen clustering setting on the held-out validation set.

Step 1: Splitting data into discovery and validation sets
---------------------------------------------------------

Before running any analysis, we split the dataset into two equal halves: a discovery set and a validation set. The
discovery set is used to fit and select between clustering settings. The validation set is held out and only used in the
final validation step.

Here is the configuration YAML file for splitting:

.. collapse:: split_data_to_discovery_and_validation.yaml

        .. highlight:: yaml
        .. code-block:: yaml

              definitions:
                datasets:
                  d1:
                    format: AIRR
                    params:
                      path: simulated_highly_similar_sequences.tsv
                      is_repertoire: false
              instructions:
                split_dataset1:
                  type: SplitDataset
                  dataset: d1
                  split_config:
                    split_count: 1
                    split_strategy: random
                    training_percentage: 0.5

To run this from the command line with immuneML installed, run:

.. code-block:: bash

    immune-ml split_data_to_discovery_and_validation.yaml ./data/

The resulting split datasets will be located under ``./data/split_dataset1/train/`` (discovery) and
``./data/split_dataset1/test/`` (validation).

Step 2: Clustering analysis on the discovery set
-------------------------------------------------

Next, we run clustering analysis on the discovery set. We will use
k-mer frequency encoding to represent the sequences, followed by PCA for dimensionality reduction,
and KMeans clustering with k=15 and k=30.

As in principle we do not know in advance which combination of encoding, dimensionality reduction, and clustering hyperparameters
will best recover the underlying structure, immuneML evaluates multiple *clustering settings* (clustering approaches) on the discovery data.
Each setting is a combination of an encoding, an optional dimensionality reduction step, and a clustering algorithm.

To choose the optimal clustering setting, we will perform the following
analysis on discovery data:

1. We will generate random subsets of the data without replacement and fit the clustering settings on each subset.
   Then, we will evaluate the clustering results using different clustering metrics (both internal and external if
   labels are available) and report the variability of the metrics across the subsets.

2. We will split the discovery data into two and measure how stable the clustering settings are across the two subsets.
   We will then repeat this for different random splits of the discovery data to get a robust estimate of clustering stability.

Clustering stability is one of the measures that can be used to inform the selection of the clustering setting.
Referring to `Liu and colleagues (2022) <https://onlinelibrary.wiley.com/doi/abs/10.1002/wics.1575>`_:

  "Stability measures capture how well partitions and clusters are preserved under perturbations to the original dataset.
  The underlying premise is that a good clustering of the data will be reproduced over an ensemble of perturbed datasets that are nearly
  identical to the original data. Stability measures the quality of perservation of clustering solutions across perturbed datasets."

To measure the stability of the clustering setting across the two subsets, immuneML implements the following procedure:
- The clustering setting is fit on the first subset, resulting in concrete cluster assignments for each data point in that subset.
- The clustering setting is fit on the second subset independently, resulting in cluster assignments for the second subset.
- A supervised classifier (which depends on the clustering algorithm used) is trained on the data from the first subset with the cluster assignments as labels. The cluster assignments for the second subset are then predicted using this classifier.
- Finally, the predicted cluster assignments for the second subset are compared to the actual cluster assignments obtained by fitting the clustering setting on the second subset using adjusted Rand index.

As this is repeated many times for different random splits of the discovery data, immuneML reports the distribution of adjusted
Rand index values across the splits, which indicates how stable the clustering setting is. This procedure follows the
procedure by `Lange et al. (2004) <https://direct.mit.edu/neco/article/16/6/1299-1323/6841>`_, with the difference that
immuneML uses adjusted Rand index to compute the similarity between cluster assignments. The review by Liu et al. (2022)
reviews this and other methods for measuring clustering stability.

In this tutorial, we will use the following settings:

.. collapse:: highly_similar_immune_data_clustering.yaml

        .. highlight:: yaml
        .. code-block:: yaml

              definitions:
                datasets:
                  d1:
                    format: AIRR
                    params:
                      is_repertoire: False
                      organism: human
                      paired: false
                      path: data/split_dataset1/train/subset_d1_train.tsv
                      dataset_file: data/split_dataset1/train/subset_d1_train.yaml
                encodings:
                  kmer: KmerFrequency
                ml_methods:
                  kmeans15:
                    KMeans:
                      n_clusters: 15
                  kmeans30:
                    KMeans:
                      n_clusters: 30
                  pca:
                    PCA:
                      n_components: 2
                reports:
                  dim_reduction_plot:
                    DimensionalityReduction:
                      dim_red_method:
                        PCA:
                          n_components: 2
                      label: motif
                  external_label_metric_summary: ExternalLabelMetricHeatmap
                  cluster_vis: ClusteringVisualization
                  external_labels_summary:
                    ExternalLabelClusterSummary:
                      external_labels: [motif]
              instructions:
                clustering_instruction:
                  clustering_settings:
                  - dim_reduction: pca
                    encoding: kmer
                    method: kmeans15
                  - dim_reduction: pca
                    encoding: kmer
                    method: kmeans30
                  dataset: d1
                  labels:
                  - motif
                  metrics:
                  - adjusted_mutual_info_score
                  - adjusted_rand_score
                  - silhouette_score
                  number_of_processes: 8
                  reports:
                  - dim_reduction_plot
                  - cluster_vis
                  - external_labels_summary
                  - external_label_metric_summary
                  stability_config:
                    split_count: 5
                    random_seed: 12
                  sample_config:
                    split_count: 5
                    percentage: 0.8
                    random_seed: 12
                  type: Clustering

To run the clustering analysis from the command line with immuneML installed, run:

.. code-block:: bash

    immune-ml highly_similar_immune_data_clustering.yaml ./clustering_results/

This will generate a report with the clustering results in the specified directory. To explore the results,
open the ``index.html`` file in ``./clustering_results/``.

Step 3: Choosing the best clustering setting
--------------------------------------------

After the clustering analysis completes, inspect the results to choose which clustering setting to validate.
The HTML report provides several views to support this decision:

- **Stability boxplot**: Shows the distribution of adjusted Rand index scores across repeated random splits of the
  discovery data for each clustering setting. A higher and less variable ARI indicates a more stable clustering.
- **External metric heatmaps**: Show how well each clustering setting recovers known labels (here: ``motif``).
  High adjusted mutual information or adjusted Rand index against the known motif labels suggests the clustering
  captures biologically meaningful structure.
- **Internal metric boxplots**: Show the silhouette score across subsamples for each setting, reflecting internal
  cluster compactness and separation.
- **Cluster visualisation**: PCA scatter plots coloured by cluster assignment and by external label, allowing
  visual inspection of how well clusters separate.

In this tutorial, the setting ``kmer + PCA + KMeans(k=15)`` shows higher stability and better agreement with
the ground-truth motif labels compared to ``KMeans(k=30)``. This is consistent with the dataset containing
exactly 15 distinct motifs. We therefore select ``kmer_pca_kmeans15`` for validation.

.. note::

  In practice, there will be no one concrete label that we can so easily compare against. It could be many different
  labels that could guide the clustering, some measured (e.g., gene usage, specificity, batch), and some not measured
  (e.g., HLA). Then, we can compare the clustering results across multiple different biologically plausible labels
  that have been measured and reported in the specific dataset.

The fitted model for the chosen setting is stored under
``clustering_results/clustering_instruction/refitted_best_settings/kmer_pca_kmeans15/kmer_pca_kmeans15.zip``.
This path is used in the next step.

Step 4: Validating the chosen clustering setting
-------------------------------------------------

Following `Ullmann and colleagues (2023) <https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1444>`_,
immuneML supports two types of clustering validation:

- **Method-based validation**: The same preprocessing, encoding, dimensionality reduction and clustering (clustering
  approach/setting) is applied independently
  to the validation data. The resulting cluster assignments are compared to any available external labels. The overall
  results are compared to the results on discovery data.
- **Result-based validation**: A supervised classifier trained on the discovery cluster assignments is used to
  predict cluster membership for the validation data. This assesses whether the cluster structure identified on
  the discovery set transfers to unseen data.

In this tutorial we run method-based validation to confirm that the ``kmer_pca_kmeans15`` setting produces
consistent clusters on the held-out validation set.

Here is the YAML configuration for validation:

.. collapse:: validate_clustering.yaml

        .. highlight:: yaml
        .. code-block:: yaml

              definitions:
                datasets:
                  d1:
                    format: AIRR
                    params:
                      is_repertoire: False
                      organism: human
                      paired: false
                      path: data/split_dataset1/test/subset_d1_test.tsv
                      dataset_file: data/split_dataset1/test/subset_d1_test.yaml
                reports:
                  dim_reduction_plot:
                    DimensionalityReduction:
                      dim_red_method:
                        PCA:
                          n_components: 2
                      label: motif
                  external_label_metric_summary: ExternalLabelMetricHeatmap
                  cluster_vis:
                    ClusteringVisualization:
                      dim_red_method:
                        PCA:
                          n_components: 2
                  external_labels_summary:
                    ExternalLabelClusterSummary:
                      external_labels: [motif]
              instructions:
                validate_clustering:
                  type: ValidateClustering
                  clustering_config_path: clustering_results/clustering_instruction/refitted_best_settings/kmer_pca_kmeans15/kmer_pca_kmeans15.zip
                  dataset: d1
                  labels:
                    - motif
                  metrics:
                    - adjusted_mutual_info_score
                    - adjusted_rand_score
                    - silhouette_score
                  number_of_processes: 8
                  reports:
                    - dim_reduction_plot
                    - cluster_vis
                    - external_labels_summary
                    - external_label_metric_summary
                  validation_type: ['method_based']

To run the validation from the command line with immuneML installed, run:

.. code-block:: bash

    immune-ml validate_clustering.yaml ./clustering_validation/

The results will be available in ``./clustering_validation/``. Open ``index.html`` to explore the validation report,
which includes cluster visualisations, external label heatmaps, and metric scores for the validation set. Comparing
these to the discovery results shows whether the clustering generalises to unseen sequences.
