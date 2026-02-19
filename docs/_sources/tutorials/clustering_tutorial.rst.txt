How to perform clustering analysis
===================================

In this tutorial, we will generate a synthetic dataset and perform clustering analysis on it.

Step 1: Creating a dataset
----------------------------

First, we will create a synthetic dataset using LIgO tool from immuneML. It generates immune receptor sequences using
Olga and simulates an immune event by implanting a list of k-mers. We will create a dataset with 100 sequences,
where 50 will contain signal1 (meaning they will have either AAA or GGG) and 50 will not contain the signal.

Here is the configuration yaml file:

.. collapse:: ligo_complete_specification.yaml

        .. highlight:: yaml
        .. code-block:: yaml

          definitions:
            motifs:
              motif1:
                seed: AAA
              motif2:
                seed: GGG
            signals:
              signal1:
                motifs: [motif1, motif2]
            simulations:
              sim1:
                is_repertoire: false # the simulation is on the sequence level (nor repertoire level)
                paired: false # we are simulating single-chain sequences
                sequence_type: amino_acid
                simulation_strategy: Implanting # how to simulate the signals
                remove_seqs_with_signals: true # remove signal-specific AIRs from the background
                sim_items:
                  sim_item: # group of AIRs with the same parameters
                    AIRR1:
                      signals:
                        signal1: 1 # all sequences in this group will have signal1
                      number_of_examples: 50 # simulate 50 sequences
                      generative_model: # how to generate background AIRs
                        default_model_name: humanTRB # use default model
                        type: OLGA # use OLGA for background simulation
                    AIRR2: # another set of sequences, but with different parameters
                      signals: {} # no signals here
                      number_of_examples: 50
                      generative_model:
                        default_model_name: humanTRB
                        type: OLGA
          instructions:
            my_sim_inst:
              export_p_gens: false
              max_iterations: 100
              number_of_processes: 4
              sequence_batch_size: 1000
              simulation: sim1
              type: LigoSim

To run this analysis from the command line with immuneML installed, run:

.. code-block:: bash

    immune-ml ligo_complete_specification.yaml ./simulated_dataset/

Step 2: Clustering analysis
----------------------------

To perform the clustering, we will use KmerFrequencyEncoding, PCA and KMeans algorithms from immuneML and scikit-learn.
We will split the data into discovery and validation set, where the discovery set will be used to fit the clustering model,
and the resulting clustering will be validated on the validation set.

As we do not know the optimal way to represent and cluster the data in advance, we will try out different combinations
of encoding, dimensionality reduction (optional) and clustering algorithms with corresponding hyperparameters. These
combinations we will call clustering settings. To choose the optimal clustering setting, we will perform the following
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
- A supervised classifier (which depends on the clustering algorithm used) is trained on the data from the first subset
  with the cluster assignments as labels. The cluster assignments for the second subset are then predicted using this classifier.
- Finally, the predicted cluster assignments for the second subset are compared to the actual cluster assignments obtained
  by fitting the clustering setting on the second subset using adjusted Rand index.

As this is repeated many times for different random splits of the discovery data, immuneML reports the distribution of adjusted
Rand index values across the splits, which indicates how stable the clustering setting is. This procedure follows the
procedure by `Lange et al. (2004) <https://direct.mit.edu/neco/article/16/6/1299-1323/6841>`_, with the difference that
immuneML uses adjusted Rand index to compute the similarity between cluster assignments. The review by Liu et al. (2022)
reviews this and other methods for measuring clustering stability.

In this tutorial, we will use the following settings:

.. collapse:: clustering_analysis.yaml

        .. highlight:: yaml
        .. code-block:: yaml

                definitions:
                  datasets:
                    d1:
                      format: AIRR
                      params:
                        path: simulated_dataset/simulated_dataset.tsv # paths to files from the previous step
                        dataset_file: simulated_dataset/simulated_dataset.yaml
                  encodings:
                    kmer: KmerFrequency # we encode the sequences using k-mer frequencies
                  ml_methods:
                    kmeans2: # we try out kmeans with k=2
                      KMeans:
                        n_clusters: 2
                    kmeans3: # and k=3
                      KMeans:
                        n_clusters: 3
                    pca:
                      PCA:
                        n_components: 4
                  reports:
                    rep1: # this is how we will visualize the data
                      DimensionalityReduction:
                        dim_red_method:
                          PCA:
                            n_components: 2
                        label: signal1 # we will color the graph by the signal we implanted
                    cluster_vis: # this will visualize clustering results
                      ClusteringVisualization: # plot a scatter plot of dim-reduced data and color the points by cluster assignments
                        dim_red_method:
                          KernelPCA: # here we can use any dimensionality reduction method supported in immuneML (see docs)
                            n_components: 2
                            kernel: rbf
                    stability: # for each split, assess how well the clusters from discovery data correspond to validation data (see docs)
                      ClusteringStabilityReport:
                        metric: adjusted_rand_score
                    external_labels_summary: # show heatmap of how cluster assignments correspond to external labels
                      ExternalLabelClusterSummary:
                        external_labels: [signal1]
                instructions:
                  clustering_instruction_with_ligo_data:
                    clustering_settings: # what combinations of encoding+dim_reduction+clustering we want to try
                    - encoding: kmer
                      method: kmeans2
                    - dim_reduction: pca
                      encoding: kmer
                      method: kmeans3
                    dataset: d1
                    labels: # here we list external labels we want to compare against if available
                    - signal1
                    metrics: # list metrics we want to use, both internal, and external (if labels are available)
                    - adjusted_rand_score
                    - adjusted_mutual_info_score
                    - silhouette_score
                    - calinski_harabasz_score
                    number_of_processes: 4
                    reports:
                    - rep1
                    - stability
                    - external_labels_summary
                    - cluster_vis
                    split_config: # we want to repeat the analysis on different splits of the data to assess stability of the results
                      split_count: 2
                      split_strategy: random # the splits will be random
                      training_percentage: 0.5 # we will use 50% of the data for discovery and 50% for validation
                    type: Clustering
                    validation_type: # the type of validation we want to perform [here we do both]
                    - result_based
                    - method_based

To run the clustering analysis from the command line with immuneML installed, run:

.. code-block:: bash

    immune-ml clustering_analysis.yaml ./clustering_results/

This will generate a report with the clustering results in the specified directory. To explore the results, see the
index.html file in output directory.

Once the analysis is done, we can explore the results and choose the optimal clustering setting. The next step is
to validate the chosen clustering setting on the validation data.

Step 3: Validation of clustering results
------------------------------------------

Following the paper by `Ullmann and colleagues (2023) <https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1444>`_,
immuneML supports two types of validation: method-based and result-based. In method-based validation, we perform the same
preprocessing+encoding+clustering on discovery and validation sets and compare the results. In result-based validation, we
fit a supervised classifier to the clusters determined on the discovery dataset and use it to predict the clustering
on the validation data, which shows if the clustering result itself is useful for validation data.