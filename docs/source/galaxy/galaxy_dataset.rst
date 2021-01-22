How to make an immuneML dataset in Galaxy
=========================================

The ‘Create dataset’ Galaxy tool allows users to import data from various formats and create immuneML datasets in Galaxy.
Before creating a dataset, the relevant data files must first be uploaded to the Galaxy interface. This can be done either
by uploading files from your local computer (use the 'Upload file' tool under the 'Get local data' menu), or by fetching
remote data from the iReceptor Plus Gateway or VDJdb (see :ref:`How to import remote AIRR datasets in Galaxy`).

The imported immuneML dataset is stored in a Galaxy collection, which will appear as a history item on the right side of the screen.
Such Galaxy collections can later be selected as input to other tools.

The tool has a simple and an advanced interface. The simple interface is fully button-based, and relies
on default settings for importing datasets. The advanced interface gives full control over import settings through a YAML
specification. In most cases, the simple interface will suffice.


immuneML datasets
-----------------
There exist three types of datasets in immuneML:

- **RepertoireDatasets** should be used when making predictions per repertoire, such as predicting a disease state.

- **SequenceDatasets** should be used when predicting values for single immune receptor chains, such as antigen specificity.

- **ReceptorDatasets** are the paired variant of SequenceDatasets, and should be used to make a prediction for each receptor chain pair.


In order to use a dataset for training ML classifiers, the metadata, which contains prediction labels, needs to be available.
For RepertoireDatasets, the metadata is supplied through a metadata file. The metadata file is a .csv file which contains
one repertoire (filename) per row, and the metadata labels for that repertoire. For more details on structuring the metadata file, see
:ref:`What should the metadata file look like?`. Note that only the Repertoire files that are present in the metadata file
will be imported.
For Sequence- and ReceptorDatasets the metadata should be available in the columns of the sequence data files. For example,
VDJdb files contain columns named 'Epitope', 'Epitope gene' and 'Epitope species'. These columns can be specified to serve
as metadata columns.


Using the simple 'Create dataset' interface
-------------------------------------------

In the simple interface the user has to select an input file format, dataset type and a list of data files to use.
For RepertoireDatasets, a metadata file must be selected from the history, whereas for Sequence- and ReceptorDatasets
the names of the columns containing metadata must be specified. The names of the metadata columns are in later
analyses available as labels for the Sequence- and ReceptorDatasets.

In subsequent YAML-based analyses, the dataset created through the simple interface should be specified like this:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_analysis_dataset: # user-defined dataset name
          format: Pickle
          params:
            path: dataset.iml_dataset


Using the advanced 'Create dataset' interface
---------------------------------------------

When using the advanced interface of the 'Create dataset' tool, a YAML specification and a list of files must be filled in.
The list of selected files should contain all data files to be imported, and additionally in the
case of a RepertoireDataset a metadata file.

The YAML specification describes how the dataset should be created from the supplied files. See :ref:`YAML specification`
for more details on writing a YAML specification file. For this tool, one :ref:`Dataset <Datasets>` must be specified
under definitions, and the :ref:`DatasetExport` instruction must be used.

The DatasetExport instruction can here only be used with one dataset (as defined under **definitions**) and one export format.
Furthermore, the **path** parameter does not need to be set. Otherwise, the specification is written the same as when running immuneML locally.

A complete YAML specification for a RepertoireDataset could look like this:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_repertoire_dataset: # user-defined dataset name
          format: VDJdb
          params:
            is_repertoire: True # import a RepertoireDataset
            metadata_file: metadata.csv # the metadata file is identified by name
            # other import parameters may be specified here
    instructions:
      my_dataset_generation_instruction: # user-defined instruction name
          type: DatasetExport
          datasets: # specify the dataset defined above
              - my_repertoire_dataset
          export_formats:
          # only one format can be specified here and the dataset in this format will be
          # available as a Galaxy collection afterwards
              - Pickle # Can be AIRR (human-readable) or Pickle (recommended)

Alternatively, for a ReceptorDataset the complete YAML specification may look like this:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_receptor_dataset: # user-defined dataset name
          format: VDJdb
          params:
            is_repertoire: False
            paired: True # if True, import ReceptorDataset. If False, import SequenceDataset
            receptor_chains: TRA_TRB # choose from TRA_TRB, TRG_TRD, IGH_IGL and IGH_IGK
            metadata_column_mapping: # VDJdb name: immuneML name
              # import VDJdb columns Epitope, Epitope gene and Epitope species, and save them
              # in metadata fields epitope, epitope_gene and epitope_species which can be used as labels
              Epitope: epitope
              Epitope gene: epitope_gene
              Epitope species: epitope_species
            # other import parameters may be specified here
    instructions:
      my_dataset_generation_instruction: # user-defined instruction name
          type: DatasetExport
          datasets: # specify the dataset defined above
              - my_receptor_dataset
          export_formats:
          # only one format can be specified here and the dataset in this format will be
          # available as a Galaxy collection afterwards
              - Pickle # Can be AIRR (human-readable) or Pickle (recommended)

Note that the export format specified here will determine how dataset import should be defined in the subsequent
YAML specifications for other immuneML Galaxy tools ('Run immuneML with YAML specification' and 'Simulate events in an immune
dataset'). The recommended format is Pickle, as it is easiest to specify dataset import from Pickle format.
If Pickle is chosen as the export format, the dataset definition for subsequent analyses will look like this:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_analysis_dataset: # user-defined dataset name
          format: Pickle
          params:
            # note that my_dataset is the name given earlier in the 'Create dataset' YAML
            path: my_dataset.iml_dataset

Alternatively, AIRR format may be specified as it is human-readable. When AIRR format is used, all relevant import
parameters need to be specified in subsequent analyses:

.. indent with spaces
.. code-block:: yaml

    definitions:
      datasets:
        my_analysis_dataset: # user-defined dataset name
          format: AIRR
          params:
            # the same value for is_repertoire and metadata_file must be used as in the first YAML
            is_repertoire: True
            metadata_file: metadata.csv
            # other import parameters may be specified here

