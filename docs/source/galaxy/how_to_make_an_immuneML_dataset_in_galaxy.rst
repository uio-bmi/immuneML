How to make an immuneML dataset in Galaxy
=========================================

immuneML dataset tool allows users to create an immuneML dataset in Galaxy.
From immune repertoire files in MiXCR, Adaptive Biotechnologies or AIRR format or
receptor files as extracted from VDJdb, a Galaxy collection (a list of immuneML-imported
files) is created and can be used as input for the immuneML wrapper tool.

The tool has three input fields:

1. YAML specification,

2. Metadata file (optional)

3. A list of files to import to a dataset.

YAML specification defines how the dataset should be created from supplied files.
It has the following format:

.. highlight:: yaml
.. code-block:: yaml

  d1: # the name of the dataset to create
    format: RandomRepertoireDataset # format of the dataset
    params: # import parameters
      repertoire_count: 100 # number of repertoires to be generated
      sequence_count_probabilities: # the probabilities have to sum to 1
        100: 0.5 # the probability that any repertoire will have 100 sequences
        120: 0.5 # the probability that any repertoire will have 120 sequences
      sequence_length_probabilities: # the probabilities have to sum to 1
        12: 0.33 # the probability that any sequence will contain 12 amino acids
        14: 0.33 # the probability that any sequence will contain 14 amino acids
        15: 0.33 # the probability that any sequence will contain 15 amino acids

In contrast to the specification used for full immuneML runs, the difference is that
no result paths are provided as input to the tool, as they are defined by Galaxy.
The list of all parameters and possible values is provided under :ref:`Datasets` in :ref:`Specification`.

Metadata file field is used when creating a dataset consisting of immune repertoires
and describes the metadata information for one repertoire per row. For the format of
the metadata file, see :ref:`What should the metadata file look like?`. For datasets consisting of immune receptor or receptor
sequences, the metadata information is provided directly in the files including receptors
as separate fields (e.g. epitope field in VDJdb format).

The third field includes all data files. For repertoire datasets, a repertoire file
corresponding to each metadata row will be imported (if a repertoire was specified in
the metadata but the corresponding repertoire file is not present, the tool will raise
an error). For receptor or receptor sequence datasets, metadata file is not used, but
instead all data from the corresponding format is imported along with its metadata.
