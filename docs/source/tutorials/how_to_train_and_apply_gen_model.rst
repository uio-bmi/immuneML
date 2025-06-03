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

Full Training Example with VAE
---------------------------------


Using Trained VAE to Generate New Sequences
-----------------------------------------------

