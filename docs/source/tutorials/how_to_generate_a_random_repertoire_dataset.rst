How to generate a random immune receptor repertoire dataset
==============================================================

Random immune receptor repertoire datasets (short: random repertoire datasets) can be used to quickly try out some immuneML functionalities, and may also be
used as a baseline when comparing different machine learning models (benchmarking, see Weber et al., Bioinformatics,
https://doi.org/10.1093/bioinformatics/btaa158). A random repertoire dataset consists of randomly generated amino acid sequences (amino acids are
chosen from a uniform distribution). The number of repertoires, number of sequences per repertoire, sequence lengths and optional labels can be
specified by the user.

Specifying a random repertoire dataset as input dataset in the YAML specification
------------------------------------------------------------------------------------

Alternatively to loading an existing dataset into immuneML, it is possible to specify a random repertoire dataset as an input dataset in the YAML
specification. This random repertoire dataset will be generated on the fly when running an immuneML analysis.

The parameters for generating a random repertoire dataset are specified under definitions/datasets in the YAML specification:

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

For the sequence count probabilities, sequence length probabilities and any custom labels multiple values can be specified, together with the
probability that each value will occur in the repertoire. These probability values must in all cases sum to 1.

It is also possible to export the generated random repertoire dataset to AIRR or Pickle format. This can be done by exporting the generated dataset
through the DatasetGeneration instruction. The generated dataset can subsequently be used for other analyses or machine learning. A complete YAML
specification for random repertoire generation and export is given below:

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


Generate a random repertoire dataset in the code
-------------------------------------------------

For developers, it is also possible to generate a random repertoire dataset directly inside the code. To do this, use the RandomDatasetGenerator
class, located in the package simulation.dataset_generation. The method generate_repertoire_dataset() uses the same parameters as described above,
and returns a RepertoireDataset object. Here is a code example:

.. highlight:: python
.. code-block:: python

  dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=100,
                                                               sequence_count_probabilities={100: 0.5, 120: 0.5},
                                                               sequence_length_probabilities={12: 0.33, 14: 0.33, 15: 0.33},
                                                               labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                               path=path)

