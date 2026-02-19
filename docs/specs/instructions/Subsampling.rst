


Subsampling is an instruction that subsamples a given dataset and creates multiple smaller dataset according to the
parameters provided.

**Specification arguments:**

- dataset (str): original dataset which will be used as a basis for subsampling

- subsampled_dataset_sizes (list): a list of dataset sizes (number of examples) each subsampled dataset should have

- subsampled_repertoire_size (int): the number of sequences to keep per repertoire (or None if all sequences should
  be kept) if dataset is a RepertoireDataset; otherwise, this argument is ignored.

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


