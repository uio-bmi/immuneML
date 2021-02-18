How to import data into immuneML
==================================

The first step of any immuneML analysis is to import the dataset that will be used. There exist three types of datasets in immuneML:

- **repertoire datasets** should be used when making predictions per repertoire, such as predicting a disease state.
  When importing a repertoire dataset, you should create a :ref:`metadata file <What should the metadata file look like?>`.

- **sequence datasets** should be used when predicting values for single immune receptor chains, such as antigen specificity.

- **receptor datasets** are the paired variant of sequence datasets, and should be used to make a prediction for each receptor chain pair.

A broad range of different import formats can be specified, including AIRR, MiXCR, VDJdb, ImmunoSEQ (Adaptive Biotechnologies),
10xGenomics, OLGA and IGoR. For the complete list of supported data formats, and extensive documentation see :ref:`Datasets`.
If you are using a custom format, or your preferred format is not yet supported, any type of tabular file can also be imported
using :ref:`Generic` import. When possible, using format-specific importers is preferred over Generic import, as they require
less options to be set and might take care of automatic reformatting of certain fields.

Alternatively to importing data from files, it is also possible to generate datasets containing random immune receptor sequences on the fly,
see :ref:`How to generate a random sequence, receptor or repertoire dataset`.


What should the metadata file look like?
------------------------------------------

The metadata file is a simple .csv file describing metadata fields for a repertoire dataset.
Metadata files are only used for repertoire datasets, for receptor and sequence datasets the metadata information should be defined as additional
columns in the same file that contains the sequences.

In case of repertoire datasets, each repertoire is represented by one file in the given format (e.g., AIRR/MiXCR/Adaptive).
For all repertoires in one dataset, a single metadata file should be defined containing the following columns:

.. image:: ../_static/images/metadata.png

The columns :code:`filename` and :code:`subject_id` are mandatory. Other columns may be defined by the user.
There are no restrictions as to what type of information these columns should represent, but typically they will represent
diseases, HLA, age or sex. These columns can be used as a prediction target (also known as :code:`labels`) when training ML models.
When writing a :ref:`YAML specification`, the :code:`labels` are defined by using the same name as the user-defined column(s) in the metadata file.


YAML specification for importing data from files
-------------------------------------------------

Data import must be defined as a part of the YAML specification. First, we choose a name which will be used to refer to the dataset in the subsequent analyses:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        ... # here, format and input parameters will be specified

The name is defined by the user. It can consist of letters, numbers and underscores.

Under the dataset name key, the :code:`format` of the data must be specified, as well as additional parameters under a key named :code:`params`.
Under :code:`format`, any of the formats listed under :ref:`Datasets` may be filled in. Under :code:`params`, the parameter :code:`path` is always
required when importing data from files. All the files must be stored in a single folder, and this folder must set through the
parameter :code:`path`.

Here is an incomplete example specification using AIRR format:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          path: path/to/data/
          ... # other import parameters will be specified here




Specifying params for repertoire dataset import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, it is assumed that a repertoire dataset should be imported. In this case, the path to the :code:`metadata_file`
must be specified. The metadata file is a .csv file which contains one repertoire (filename) per row, and the metadata
labels for that repertoire. These metadata labels can be used as a prediction target when training ML models.
For more details on structuring the metadata file, see :ref:`What should the metadata file look like?`.
Note that only the repertoire files that are present in the metadata file will be imported.

Other parameters that are specific to the format may be specified under :code:`params` as well, and are explained in more detail for each format
under :ref:`Datasets`.

A complete specification for importing a repertoire dataset from AIRR format with default parameters may look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          # required parameters
          path: path/to/data/
          metadata_file: path/to/metadata.csv
          # is_repertoire is by default True, and may be omitted
          is_repertoire: True
          # Other parameters specific to AIRR data may be specified here


Specifying params for receptor or sequence dataset import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to import a sequence or receptor dataset, set the parameter :code:`is_repertoire` to false, and set :code:`paired` to either false (sequence dataset)
or true (receptor dataset). For sequence and receptor dataset, metadata labels must be specified directly as columns in the input files.
These metadata labels can be used as a prediction target when training ML models. For example, a column 'binding' can be added, which may have values 'true' and 'false'.
The metadata labels are specified through parameter :code:`metadata_column_mapping`, which is a mapping from the names of the columns in
the file to the names that will be used internally in immuneML (for example: when specifying :code:`labels` in the :ref:`TrainMLModel` instruction).
It is recommended that the immuneML-internal names contain only lowercase letters, numbers and underscores.

A complete specification for importing a sequence dataset from AIRR format with default parameters may look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          # required parameters
          path: path/to/data/
          is_repertoire: false
          paired: false # must be true for receptor dataset and false for sequence datasets
          metadata_column_mapping: # metadata column mapping AIRR: immuneML
            binding: binding # the names could just be the same
            Epitope.gene: epitope_gene # if the column name contains undesired characters, it may be renamed for internal use
          # Other parameters specific to AIRR data may be specified here

For receptor datasets, the additional parameter :code:`receptor_chains` needs to be set, which determines the type
of chain pair that should be imported. The resulting specification may look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          # required parameters
          path: path/to/data/
          is_repertoire: false
          paired: true # must be true for receptor dataset and False for sequence datasets
          receptor_chains: TRA_TRB # choose from TRA_TRB, TRG_TRD, IGH_IGL and IGH_IGK
          metadata_column_mapping: # metadata column mapping AIRR: immuneML
            binding: binding # the names could just be the same
            Epitope.gene: epitope_gene # if the column name contains undesired characters, it may be renamed for internal use
          # Other parameters specific to AIRR data may be specified here


Importing previously generated immuneML datasets
------------------------------------------------

When you import a dataset into immuneML for the first time, it is converted to an optimized binary format,
which speeds up the analysis. The main resulting file has an `.iml_dataset` extension, and may be accompanied
by several other `.pickle` and `.npy` files. When running immuneML locally, you can by default find these immuneML
dataset files in the folder 'datasets', which is located in the main output folder of your analysis.

Some instructions (:ref:`Simulation`, :ref:`DatasetExport`, :ref:`SubSampling`) also explicitly export binarized immuneML
datasets when selecting 'Pickle' as the export format.

These `.iml_dataset` files can later be imported easily and with few parameters, and importing from `.iml_dataset` is
also faster than importing from other data formats. A YAML specification for Pickle data import is shown below.
Important note: Pickle files might not be compatible between different immuneML (sub)versions.

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset:
        format: Pickle
        params:
          path: path/to/dataset.iml_dataset
          # specifying a metadata_file is optional, it will update the dataset using this new metadata.
          metadata_file: path/to/metadata.csv

