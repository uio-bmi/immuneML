How to train a generative model
========================================

This tutorial provides a practical introduction for AIRR researchers interested in training generative machine
learning models on immune receptor sequences using immuneML and :ref:`TrainGenModel` instruction.

Choosing a Dataset
---------------------

To train a generative model, you need a dataset of immune receptor sequences. The sequences should be in any format
supported by immuneML, such as AIRR or VDJdb formats. See :ref:`Dataset parameters` for the full list of supported formats
and necessary parameters.

Overview of Generative Models in immuneML
---------------------------------------------

immuneML supports several approaches for training generative models:

- positional weight matrices (:ref:`PWM`),
- LSTM-based generative models (:ref:`SimpleLSTM`),
- Variational Autoencoders (:ref:`SimpleVAE`),
- SoNNia model (:ref:`SoNNia`).

See the documentation for each model for details on how to configure them. Some require almost no parameters, while
others allow greater flexibility and customization.

Reports to Analyze the Results
-----------------------------------

immuneML provides built-in reports to inspect and evaluate generative models - either directly or in combination
with different feature representations:

- :ref:`PWMSummary` showing probabilities of generated sequences having different lengths and PWMs for each length,
- :ref:`VAESummary` showing the latent space after reducing the dimensionality to 2 dimensions,
  histogram for each latent dimension, loss per epoch.
- :ref:`AminoAcidFrequencyDistribution` showing the distribution of amino acids in the generated vs original sequences,
- :ref:`SequenceLengthDistribution` showing the distribution of sequence lengths in the generated vs original sequences,
- :ref:`FeatureComparison` comparing the generated sequences with the original dataset using different encodings
  (e.g., k-mer frequencies (:ref:`KmerFrequency`), or protein embeddings (:ref:`ESMC`, :ref:`TCRBert`, :ref:`ProtT5`)).
- :ref:`DimensionalityReduction` to compare encoded sequences after applying dimensionality reduction
  (see :ref:`***Dimensionality reduction methods***`) and coloring the points by labels (e.g., generated vs original sequences).

Full Training Example with LSTM
---------------------------------

To train an LSTM, the following YAML configuration may be used:

.. code-block:: yaml

  definitions:
    datasets:
      dataset:
        format: AIRR
        params:
          path: original_dataset.tsv
          is_repertoire: False
          paired: False
          region_type: IMGT_CDR3
          separator: "\t"
    ml_methods:
      LSTM:
        SimpleLSTM:
          locus: beta
          sequence_type: amino_acid
          num_epochs: 20
          hidden_size: 1024
          learning_rate: 0.001
          batch_size: 100
          embed_size: 256
          temperature: 1
          num_layers: 3
          device: cpu
          region_type: IMGT_CDR3

  instructions:
    LSTM:
      type: TrainGenModel
      export_combined_dataset: True
      dataset: dataset
      method: LSTM
      gen_examples_count: 1500
      number_of_processes: 1
      training_percentage: 0.7

To explore the dataset with original and generated sequences, we could encode them using k-mer frequencies and visualize
with feature value barplots:

.. code-block:: yaml

  definitions:
    datasets:
      LSTM_dataset:
        format: AIRR
        params:
          path: dataset.tsv
          is_repertoire: False
          paired: False
          region_type: IMGT_CDR3
          separator: "\t"
          import_illegal_characters: True
    encodings:
      3mer_encoding:
        KmerFrequency:
          k: 3
          sequence_type: amino_acid
          scale_to_unit_variance: False
          scale_to_zero_mean: False
      gapped_4mer_encoding:
        KmerFrequency:
          sequence_encoding: gapped_kmer
          sequence_type: amino_acid
          k_left: 2
          k_right: 2
          min_gap: 1
          max_gap: 1
          scale_to_unit_variance: False
          scale_to_zero_mean: False
    reports:
      feature_value_barplot:
        FeatureValueBarplot:
          color_grouping_label: dataset_split
          plot_all_features: false
          plot_top_n: 25
          error_function: sem

  instructions:
    data_reports:
      type: ExploratoryAnalysis
      number_of_processes: 1
      analyses:
        LSTM_3mer_analysis:
          dataset: LSTM_dataset
          encoding: 3mer_encoding
          reports: [ feature_value_barplot ]
        LSTM_gapped_4mer_analysis:
          dataset: LSTM_dataset
          encoding: gapped_4mer_encoding
          reports: [ feature_value_barplot ]



Using Trained VAE to Generate New Sequences
-----------------------------------------------

