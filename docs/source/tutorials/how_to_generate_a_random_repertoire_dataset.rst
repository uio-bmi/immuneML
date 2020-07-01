How to generate a random immune receptor repertoire dataset
===========================================================

Random immune receptor datasets (short: random repertoire datasets) can be used to quickly try out some immuneML functionalities, and could also be
used as a baseline when comparing different machine learning models (benchmarking, see Weber et al., Bioinformatics,
https://doi.org/10.1093/bioinformatics/btaa158).

Generate random repertoire dataset from YAML together with the rest of the analysis
---------------------------------------------------------------------------------

To generate a repertoire dataset in YAML consisting of random amino acid sequences, define the desired parameters of the dataset in the datasets
section. In the same manner as an already existing dataset can be loaded into the platform, for the random dataset it is possible to define its
parameters and import the random dataset for further analysis.

DSL specification example under definitions/datasets:

.. highlight:: yaml
.. code-block:: yaml

  datasets:
    d1:
      format: RandomRepertoireDataset
      params:
        result_path: ./ # where to store the resulting dataset
        repertoire_count: 100 # number of repertoires to be generated
        sequence_count_probabilities: # the probabilities have to sum to 1
          100: 0.5 # the probability that any repertoire will have 100 sequences
          120: 0.5 # the probability that any repertoire will have 120 sequences
        sequence_length_probabilities: # the probabilities have to sum to 1
          12: 0.33 # the probability that any sequence will contain 12 amino acids
          14: 0.33 # the probability that any sequence will contain 14 amino acids
          15: 0.33 # the probability that any sequence will contain 15 amino acids
        labels: # metadata that can be used as labels, can also be empty
          HLA: # label name (the probabilities per label value have to sum to 1)
            A: 0.5 # the probability that any generated repertoire will have HLA A
            B: 0.5 # the probability that any generated repertoire will have HLA B

Note that all probabilities have to be floating point values and have to sum to 1. All parameters (repertoire_count, sequence_count_probabilities,
sequence_length_probabilities and labels) have to be specified, though their values can of course be different.

Full specification example along with an instruction to export the data in desired formats is given below. For now, the only available formats are
AIRR and Pickle (as listed below). The generated dataset can then be used for other analyses or machine learning (to try out the example, download
the YAML file):

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      d1: # dataset name to use later in the instruction
        format: RandomRepertoireDataset
        params:
          labels: {}
          repertoire_count: 100
          sequence_count_probabilities:
            100: 0.5
            120: 0.5
          sequence_length_probabilities:
            10: 1.0
          result_path: random_dataset_workflow/
  instructions:
    my_dataset_generation_instruction:
      type: DatasetGeneration
      datasets: [d1] # list of datasets to export
      formats: [AIRR, Pickle] # list of formats to export the datasets to


Generate random repertoire dataset in the code
----------------------------------------------

To do the same from the code, use the `RandomDatasetGenerator` class, located in the package simulation.dataset_generation.
The method `generate_repertoire_dataset()` will create a random repertoire dataset for the given parameters. The parameters are the same as in
section 1 of this tutorial and include the number of repertoires to generate (repertoire_count), the probability distribution of sequence count
per repertoire, the probability distribution of sequence lengths, labels with probabilities for any repertoire getting any of the values and the
path where the resulting dataset will be stored. Here is a code example:

.. highlight:: python
.. code-block:: python

  dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=100,
                                                               sequence_count_probabilities={100: 0.5, 120: 0.5},
                                                               sequence_length_probabilities={12: 0.33, 14: 0.33, 15: 0.33},
                                                               labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                               path=path)

