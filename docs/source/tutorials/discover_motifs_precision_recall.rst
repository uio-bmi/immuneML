Discovering positional motifs using precision and recall thresholds
----------------------------------------------------------------------

It is often assumed that the antigen binding status of an immune receptor (antibody/TCR) may be determined by the *presence*
of a short motif in the CDR3.
We developed a method (manuscript in preparation) for the discovery of antigen binding associated motifs with the following properties:

- Short position-specific motifs with possible gaps
- High precision for predicting antigen binding
- High generalisability to unseen data, i.e., retaining a relatively high precision on test data


Method description
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A motif with a high precision for predicting antigen binding implies that when the motif is present,
the probability that the sequence is a binder is high. One can thus iterate through every possible motif and filter
them by applying a precision threshold. However, the more 'rare' a motif is, the more likely that the motif just had
a high precision by chance (for example: a motif that occurs in only 1 binder and 0 non-binders has a perfect precision,
but may not retain high precision on unseen data). Thus, an additional recall threshold is applied to remove
rare motifs.
Our method allows the user to define a precision threshold and learn the optimal recall threshold using a training + validation set.

The method consists the following steps:

1. Splitting the data into training, validation and test sets.

2. Using the training set, find all motifs with a high training-precision.

3. Using the validation set, determine the recall threshold for which the validation-precision is still high (separate recall thresholds may be learned for motifs with different sizes).

4. Using the combined training + validation set, find all motifs exceeding the user-defined precision threshold and learned recall threshold(s).

5. Using the test set, report the precision and recall of these learned motifs.

6. Optional: use the set of learned motifs as input features for ML classifiers (e.g., :ref:`BinaryFeatureClassifier` or :ref:`LogisticRegression`) for antigen binding prediction.

Steps 2+3 are done by the report :ref:`MotifGeneralizationAnalysis`. This report exports the learned recall cutoff(s).
It is recommended to run this report using the :ref:`ExploratoryAnalysis` instruction.
Steps 4+5 are done by the :ref:`Motif` encoder. The learned recall cutoff(s) are used as input parameters. This encoder
can be used either in :ref:`ExploratoryAnalysis` or :ref:`TrainMLModel` instructions.

