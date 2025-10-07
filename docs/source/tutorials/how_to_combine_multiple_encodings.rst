How to combine multiple encodings to represent a dataset
============================================================

Sometimes it might be of interest to combine multiple encodings to represent a dataset, e.g., by combining k-mer
frequencies with V or J gene frequencies, or k-mer frequencies with certain metadata fields to try to control for
differences, e.g., in HLA types. immuneML support combining multiple encodings by using the :ref:`Composite` encoder and
this tutorial illustrates how to do this.

To illustrate this usage, we will:

- create a random repertoire dataset and assign some random metadata values to each repertoire
- create a composite encoding that combines k-mer frequencies with the metadata values
- combine the composite encoding with a logistic regression classifier to create a simple ML pipeline

Additionally, we will illustrate how to use :ref:`LogRegressionCustomPenalty` to assign different penalties to the
different parts of the composite encoding. This might be of interest when the different parts of the encoding have
different number of features or larger differences in value ranges.

To create a random dataset with randomly assigned HLA metadata values, we can use the following YAML specification:

.. code-block:: yaml

    datasets:
      dataset:
        format: RandomRepertoireDataset
        params:
          repertoire_count: 100 # number of repertoires to generate
          sequence_count_probabilities:
            10: 0.5 # probability that any repertoire would have 10 receptor sequences
            20: 0.5 # probability that any repertoire would have 20 receptor sequences
          sequence_length_probabilities:
            10: 0.5 # probability that any sequence in a repertoire is 10 a.a. long
            12: 0.5 # probability that any sequences in a repertoire is 12 a.a. long
          labels: # labels which can be used for machine learning
            disease: # the name of the label corresponding to an immune event
              True: 0.5 # probability that a repertoire is positive w.r.t. the label
              False: 0.5 # probability that a repertoire is negative w.r.t. the label
            hla:
              A1: 0.5
              A2: 0.5

To create a composite encoder that combines k-mer frequencies with the HLA metadata values, we can use the following YAML
specification:

.. code-block:: yaml

    encodings:
      kmer_freq_hla:
        Composite:
          encoders:
          - KmerFrequency:
              k: 3
          - Metadata:
              metadata_fields: [hla]

The resulting feature vector will contain the k-mer frequencies followed by the one-hot encoded HLA metadata values.

To create a logistic regression classifier that assigns different penalties to the k-mer frequencies and the HLA metadata
values, we can use the following YAML specification:

.. code-block:: yaml

  ml_methods:
    log_reg_custom_penalty:
      LogRegressionCustomPenalty:
        alpha: 1
        n_lambda: 100
        non_penalized_encodings: ['MetadataEncoder']
        random_state: 42

In this case, we do not penalize the coefficients corresponding to the HLA metadata values because they will be
few and one-hot encoded, while the k-mer frequencies will be many and continuous-valued.

Here is the complete YAML specification that combines all of the above to create a simple ML pipeline:

.. code-block:: yaml

  definitions:
    datasets:
      dataset:
        format: RandomRepertoireDataset
        params:
          repertoire_count: 100 # number of repertoires to generate
          sequence_count_probabilities:
            10: 0.5 # probability that any repertoire would have 10 receptor sequences
            20: 0.5 # probability that any repertoire would have 20 receptor sequences
          sequence_length_probabilities:
            10: 0.5 # probability that any sequence in a repertoire is 10 a.a. long
            12: 0.5 # probability that any sequences in a repertoire is 12 a.a. long
          labels: # labels which can be used for machine learning
            disease: # the name of the label corresponding to an immune event
              True: 0.5 # probability that a repertoire is positive w.r.t. the label
              False: 0.5 # probability that a repertoire is negative w.r.t. the label
            hla:
              A1: 0.5
              A2: 0.5

    encodings:
      kmer_freq_hla:
        Composite:
          encoders:
          - KmerFrequency:
              k: 3
          - Metadata:
              metadata_fields: [hla]

    ml_methods:
      log_reg_custom_penalty:
        LogRegressionCustomPenalty:
          alpha: 1
          n_lambda: 100
          non_penalized_encodings: ['MetadataEncoder']
          random_state: 42

    reports:
      coefficients:
        Coefficients:
          coefs_to_plot:
            - all
            - nonzero
            - n_largest
          n_largest:
            - 30
      performance_per_hla:
        PerformancePerLabel:
          alternative_label: hla
          metric: balanced_accuracy
          compute_for_selection: false
          compute_for_assessment: true

  instructions:
    train_classifier:
      type: TrainMLModel

      dataset: dataset
      labels: [disease]

      settings:
        - encoding: kmer_freq_hla
          ml_method: log_reg_custom_penalty

      assessment:
        split_strategy: random
        split_count: 1
        reports:
          models:                # plot the coefficients of the trained models
          - coefficients

      selection:
        split_strategy: k_fold
        split_count: 3

      optimization_metric: balanced_accuracy # the metric used for optimization
      metrics: # other metrics to compute
      - precision
      - recall
      - auc

      reports:
      - performance_per_hla

To run this example, save the above YAML specification to a file named ``composite_encoding_example.yaml``. We assume you
have installed immuneML (in a virtual environment) and can run from the console. If you haven't set it up yet,
see :ref:`Install immuneML with a package manager`.
When you have immuneML installed and environment activated, run the following command in your terminal:

.. code-block:: console

    immune-ml composite_encoding_example.yaml composite_encoding_example_output

The results can be explored from `composite_encoding_example_output/index.html`.