How to make an immuneML dataset in Galaxy
=========================================

The ‘Create dataset’ Galaxy tool allows users to import data from various formats and create immuneML datasets in Galaxy.
The imported files are stored as a Galaxy collection, which will appear as a history item on the right side of the screen.
Such Galaxy collections can later be selected as input to other tools.

The tool can be used to create Repertoire-, Sequence- or ReceptorDatasets. RepertoireDatasets should be used when making
predictions per repertoire, such as predicting a disease state. SequenceDatasets or ReceptorDatasets should be used when
predicting values for unpaired (single-chain) and paired immune receptors respectively, like antigen specificity.

The ‘Create dataset’ tool has a simple and an advanced interface. The simple interface is fully button-based, and relies
on default settings for importing datasets. The advanced interface gives full control over import settings through a YAML
specification. In most cases, using the simple interface will suffice.

Using the simple interface
--------------------------




Using the advanced interface
----------------------------


The advanced interface has two input fields:

1. A YAMl specification,

2. A list of data and metadata files to import


The YAML specification describes how the dataset should be created from the supplied files. See :ref:`YAML specification`
for more details on writing a YAML specification file. For this tool, one dataset :ref:`Datasets` must be specified under definitions, and
the :ref:`DatasetGeneration` instruction must be used.

The `path` parameter for the dataset must always be set to the current working directory (`./`), and the DatasetGeneration
instruction can here only be used with one dataset and one export format. Otherwise, the specification is written the same
as when running immuneML locally.

A complete YAML specification could look like this:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_dataset: # user-defined dataset name
          format: VDJdb
          params:
            is_repertoire: True
            metadata_file: metadata.csv # the metadata file is identified by name
            path: ./ # the path must always be the current working directory
            # other import parameters may be specified here
    instructions:
      my_dataset_generation_instruction: # user-defined instruction name
          type: DatasetGeneration # which instruction to execute
          datasets: # only one dataset can be given here
              - my_dataset
          export_formats:
          # only one format can be specified here and the dataset in this format will be
          # available as a Galaxy collection afterwards
              - AIRR


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




For this tool we need to specify a dataset under definitions, and




immuneML dataset tool allows users to create an immuneML dataset in Galaxy.
From immune repertoire files in MiXCR, Adaptive Biotechnologies or AIRR format or
receptor files as extracted from VDJdb, a Galaxy collection (a list of immuneML-imported
files) is created and can be used as input for the immuneML wrapper tool.

The advanced interface has three input fields:

1. YAML specification,

2. Metadata file (optional)

3. A list of files to import to a dataset.

YAML specification defines how the dataset should be created from supplied files. See :ref:`YAML specification` for more details on writing a YAML
specification file, specifically with :ref:`DatasetGeneration` instruction. For this Galaxy tool, the specification will include only one dataset
and only one format in which it will be exported. For instance, YAML specification could be the following:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_dataset: # dataset which to use to create a Galaxy collection
          format: AdaptiveBiotech
          params:
            metadata_file: metadata.csv
            path: ./
    instructions:
      my_dataset_generation_instruction: # user-defined instruction name
          type: DatasetGeneration # which instruction to execute
          datasets: # only one dataset can be given here
              - my_dataset
          export_formats:
          # only one format can be specified here and the dataset in this format will be
          # available as a Galaxy collection afterwards
              - AIRR

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
