


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


