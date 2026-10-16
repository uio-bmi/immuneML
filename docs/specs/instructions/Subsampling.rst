


Subsampling is an instruction that subsamples a given dataset and creates multiple smaller dataset according to the
parameters provided.

**Specification arguments:**

- dataset (str): original dataset which will be used as a basis for subsampling

- subsampled_dataset_sizes (list): a list of dataset sizes (number of examples) each subsampled dataset should have

- subsampled_repertoire_size (int): the number of sequences to keep per repertoire (or None if all sequences should
  be kept) if dataset is a RepertoireDataset; otherwise, this argument is ignored.

- label (str or dict): the label to use for class-balanced subsampling; it can be specified either just by name (in
  which case the values and positive class are inferred from the dataset), or as a dictionary with a single key
  (label name) and value ``positive_class`` to explicitly set the positive class. Required if
  ``subsampled_class_distributions`` is set; otherwise ignored. See :ref:`TrainMLModel` for more details on how
  labels are specified.

- subsampled_class_distributions (list): a list of the same length as ``subsampled_dataset_sizes``, where each
  element is a dictionary mapping the values of ``label`` to the fraction of the corresponding subsampled dataset
  that should have that class value; the fractions in each dictionary have to sum to 1. For a binary label, it is
  enough to provide the fraction for one of the two classes -- the other is set to 1 minus that fraction. When this
  argument (together with ``label``) is provided, examples are sampled per class without replacement (uniformly at
  random within each class) instead of uniformly at random across the whole dataset, so that the resulting
  subsampled dataset matches the requested class distribution. If not set, subsampling is done uniformly at random
  as before, without considering any label.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_subsampling_instruction: # user-defined name of the instruction
            type: Subsampling # which instruction to execute
            dataset: my_dataset # original dataset to be subsampled, with e.g., 300 examples
            subsampled_dataset_sizes: # how large the subsampled datasets should be, one dataset will be created for each list item
                - 200 # one subsampled dataset with 200 examples (200 repertoires if my_dataset was repertoire dataset)
                - 100 # the other subsampled dataset will have 100 examples

        my_class_balanced_subsampling_instruction: # user-defined name of the instruction
            type: Subsampling
            dataset: my_dataset
            label: my_binary_label # label to use for class-balanced subsampling
            subsampled_dataset_sizes: [500, 500, 500] # sweep over class balance at a fixed size
            subsampled_class_distributions: # one entry per item in subsampled_dataset_sizes
                - {positive: 0.1} # 50 examples with class 'positive', 450 with the other class
                - {positive: 0.3}
                - {positive: 0.5}


