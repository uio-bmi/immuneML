How to simulate an AIRR dataset in Galaxy
===================================================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML & Galaxy: simulate an AIRR dataset
   :twitter:description: See tutorials on how to simulate an AIRR dataset in Galaxy.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png



The Galaxy tool `Simulate a synthetic immune receptor or repertoire dataset <https://galaxy.immuneml.uio.no/root?tool_id=immuneml_simulate_dataset>`_ allows you to quickly make a dummy dataset.
The tool generates a SequenceDataset, ReceptorDataset or RepertoireDataset consisting of random CDR3 sequences, which could be used for benchmarking machine learning methods or encodings,
or testing out other functionalities.
The amino acids in the sequences are chosen from a uniform random distribution, and there is no underlying structure in the sequences.

You can control:

- The amount of sequences in the dataset, and in the case of a RepertoireDataset, the amount of repertoires

- The length of the generated sequences

- Labels, which can be used as a target when training ML models

Note that since these labels are randomly assigned, they do not bear any meaning and it is not possible train a ML model with high classification accuracy on this data.
Meaningful labels can be added by :ref:`simulating immune events into an existing AIRR dataset <How to simulate immune events into an existing AIRR dataset in Galaxy>`.

An example Galaxy history showing how to use this tool `can be found here <https://galaxy.immuneml.uio.no/u/immuneml/h/simulate-dataset>`_.

Creating the YAML specification
---------------------------------------------

To run this tool, the user should provide a YAML specification describing the analysis.
The YAML specification should use :ref:`RandomSequenceDataset`, :ref:`RandomReceptorDataset` or :ref:`RandomRepertoireDataset` import in combination with the :ref:`DatasetExport` instruction.
A complete example of a full YAML specification for generating a RandomRepertoireDataset is shown here:


.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      dataset: #user-defined dataset name
        format: RandomRepertoireDataset # alternatively, choose RandomSequenceDataset or RandomReceptorDataset (note they have different params)
        params:
          labels: # metadata that can be used as labels, can also be empty
            HLA: # user-defined label name
              A: 0.6 # user-defined label values, the probabilities must sum to 1
              B: 0.4
          repertoire_count: 100
          sequence_count_probabilities: # the probabilities of finding each number of sequences in a repertoire, must sum to 1
            1000: 0.5
            1200: 0.5
          sequence_length_probabilities: # the probabilities of finding each sequence length in a repertoire, must sum to 1
            12: 0.25
            13: 0.25
            14: 0.25
            15: 0.25
  instructions:
    my_dataset_export_instruction: # user-defined instruction name
      type: DatasetExport
      datasets: # specify the dataset defined above
      - dataset
      export_formats:
      # only one format can be specified here and the dataset in this format will be
      # available as a Galaxy collection afterwards
      - Pickle # Can be AIRR (human-readable) or Pickle (recommended for further Galaxy-analysis)





..
    Simulations of a repertoire and a receptor dataset are shown in the figures below.

    .. figure:: ../_static/images/simulate_immune_repertoire_dataset.png
      :width: 70%

    .. figure:: ../_static/images/simulate_immune_receptor_dataset.png
      :width: 70%



Tool output
---------------------------------------------
This Galaxy tool will produce the following history elements:

- Summary: dataset simulation: a HTML page describing general characteristics of the dataset, including the name of the dataset
  (this name should be specified when importing the dataset later in immuneML), the dataset type and size, and a link to download
  the raw data files.

- Archive: dataset simulation: a .zip file containing the complete output folder as it was produced by immuneML. This folder
  contains the output of the DatasetExport instruction including raw data files.
  Furthermore, the folder contains the complete YAML specification file for the immuneML run, the HTML output and a log file.

- Simulated immuneML dataset: Galaxy collection containing all relevant files for the new dataset.
