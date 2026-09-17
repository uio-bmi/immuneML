How to generate a dataset with random sequences
=================================================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: generate a random dataset
   :twitter:description: See tutorials on how to generate a dataset with random sequences
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


Random immune receptor sequence, immune receptor or immune repertoire datasets (short: random sequence/receptor/repertoire datasets) can be used to quickly try out some immuneML functionalities, and may also be
used as a baseline when comparing different machine learning models (benchmarking, see Weber et al., Bioinformatics,
https://doi.org/10.1093/bioinformatics/btaa158). A random sequence/receptor/repertoire dataset consists of randomly generated amino acid sequences, where the amino acids are
chosen from a uniform distribution. The dataset size, sequence lengths and optional labels can be specified by the user.

The generated dataset can then be used to train a classifier (see :ref:`How to train and assess a receptor or repertoire-level ML classifier`),
apply a classifier (see :ref:`How to apply previously trained ML models to a new dataset`), or simulate immune events (see
:ref:`Dataset simulation with LIgO`).

YAML specification of a random repertoire dataset
-------------------------------------------------

Alternatively to loading an existing dataset into immuneML, it is possible to specify a random repertoire dataset as an input dataset in the YAML
specification. This random repertoire dataset will be generated on the fly when running an immuneML analysis.

The parameters for generating a random repertoire dataset are specified under definitions/datasets in the YAML specification:

.. highlight:: yaml
.. code-block:: yaml

  datasets:
    my_dataset:
      format: RandomRepertoireDataset
      params:
        repertoire_count: 100 # number of repertoires to be generated
        sequence_count_probabilities: # the probabilities have to sum to 1
          100: 0.5 # the probability that any repertoire will have 100 sequences
          120: 0.5 # the probability that any repertoire will have 120 sequences
        sequence_length_probabilities: # the probabilities have to sum to 1
          12: 0.33 # the probability that any sequence will contain 12 amino acids
          14: 0.33 # the probability that any sequence will contain 14 amino acids
          15: 0.33 # the probability that any sequence will contain 15 amino acids
        labels: # metadata that can be used as labels, can also be empty
          HLA: # label name, any name can be chosen (the probabilities per label value have to sum to 1)
            A: 0.6 # the probability that any generated repertoire will have HLA A
            B: 0.4 # the probability that any generated repertoire will have HLA B

For the sequence count probabilities, sequence length probabilities and any custom labels multiple values can be specified, together with the
probability that each value will occur in the repertoire. These probability values must in all cases sum to 1.


YAML specification of a random sequence dataset
-----------------------------------------------

Specifying a random sequence dataset is similar to specifying a random repertoire dataset, except there are some minor differences
in the settings.

.. highlight:: yaml
.. code-block:: yaml

  datasets:
    my_dataset:
      format: RandomSequenceDataset
      params:
        sequence_count: 100 # number of receptors to be generated
        length_probabilities:
          14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
          15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
        labels: # metadata that can be used as labels, can also be empty
          binds_epitope: # label name, any name can be chosen (the probabilities per label value have to sum to 1)
            True: 0.6 # 60% of the receptors will have class True
            False: 0.4 # 40% of the receptors will have class False



YAML specification of a random receptor dataset
-----------------------------------------------

Finally, a random receptor dataset can be specified as follows:

.. highlight:: yaml
.. code-block:: yaml

  datasets:
    my_dataset:
      format: RandomReceptorDataset
      params:
        receptor_count: 100 # number of receptors to be generated
        chain_1_length_probabilities:
          14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
          15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
        chain_2_length_probabilities:
          14: 0.8
          15: 0.2
        labels: # metadata that can be used as labels, can also be empty
          binds_epitope: # label name, any name can be chosen (the probabilities per label value have to sum to 1)
            True: 0.6 # 60% of the receptors will have class True
            False: 0.4 # 40% of the receptors will have class False


Exporting a random sequence/receptor/repertoire dataset
-------------------------------------------------------

It is possible to export the generated random sequence/receptor/repertoire dataset to AIRR or ImmuneML format. This can be done by exporting the generated dataset
through the :ref:`DatasetExport` instruction. The generated dataset can subsequently be used for other analyses or machine learning. A complete YAML
specification for random repertoire generation and export is given below:

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset:
        # this is the definition for a random repertoire dataset,
        # alternatively, the definition of a random sequence/receptor dataset can be specified
        format: RandomRepertoireDataset
        params:
          labels: {}
          repertoire_count: 100
          sequence_count_probabilities:
            100: 0.5
            120: 0.5
          sequence_length_probabilities:
            10: 1.0
  instructions:
    my_dataset_export_instruction:
      type: DatasetExport
      datasets: [my_dataset] # list of datasets to export
      export_formats: [AIRR, ImmuneML] # list of formats to export the datasets to.


Generating random sequence/receptor/repertoire datasets in the code
--------------------------------------------------------------------

For developers, it is also possible to generate a random receptor/repertoire dataset directly inside the code. To do this, use the RandomDatasetGenerator
class, located in the package simulation.dataset_generation. The methods below use the same parameters as described above,
and returns a SequenceDataset, ReceptorDataset or RepertoireDataset object:

.. highlight:: python
.. code-block:: python

  repertoire_dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=100,
                                                               sequence_count_probabilities={100: 0.5, 120: 0.5},
                                                               sequence_length_probabilities={12: 0.33, 14: 0.33, 15: 0.33},
                                                               labels={"HLA": {"A": 0.5, "B": 0.5}},
                                                               path=path)

  sequence_dataset = RandomDatasetGenerator.generate_receptor_dataset(sequence_count=100,
                                                               length_probabilities={12: 0.33, 14: 0.33, 15: 0.33},
                                                               labels={"binds_epitope": {"True": 0.5, "False": 0.5}},
                                                               path=path)

  receptor_dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=100,
                                                               chain_1_length_probabilities={12: 0.33, 14: 0.33, 15: 0.33},
                                                               chain_2_length_probabilities={12: 0.33, 14: 0.33, 15: 0.33},
                                                               labels={"binds_epitope": {"True": 0.5, "False": 0.5}},
                                                               path=path)