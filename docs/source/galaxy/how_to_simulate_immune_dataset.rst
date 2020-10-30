Simulate an immune receptor or repertoire dataset
===================================================

This tool allows you to quickly make a dataset which could be used for benchmarking machine learning methods or encodings,
or testing out other functionalities. The tool generates either a repertoire or a receptor dataset consisting of CDR3 sequences. The sequences are
made of randomly chosen amino acids and there is no underlying structure in the sequences. You can control how many repertoires and receptors to
generate and the length of the receptor sequences. Additionally, you can define labels with possible values which will be randomly assigned to the
receptors or repertoires.

Simulations of a repertoire and a receptor dataset are shown in the figures below.

.. figure:: ../_static/images/simulate_immune_repertoire_dataset.png
  :width: 70%

.. figure:: ../_static/images/simulate_immune_receptor_dataset.png
  :width: 70%

The tool takes a YAML file as input and outputs a dataset collection either in Pickle or AIRR format which can then be downloaded or used as input
for other immuneML Galaxy tools.

YAML specification which creates a repertoire dataset and can be used as input for this tool is given below:

.. code-block:: yaml

  definitions:
    datasets:
      my_dataset:
        format: RandomRepertoireDataset
        params:
          repertoire_count: 100
          sequence_count_probabilities: # probabilities of given sequence counts per repertoire
            100: 0.5
            150: 0.5
          sequence_length_probabilities: # probabilities of given lengths for receptor sequences
            10: 0.5
            11: 0.2
            13: 0.3
          labels:
            my_test_label: # a label with values randomly assigned to repertoires per given probabilities
              True: 0.5
              False: 0.5
  instructions:
    my_dataset_sim_instruction:
      type: DatasetGeneration
      export_formats: [AIRR] # only one format can be specified in Galaxy
      datasets: [my_dataset] # only one dataset can be specified in Galaxy