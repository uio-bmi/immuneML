How to import data into immuneML
==================================

The first step of any immuneML analysis is to import the dataset that will be used. There exist three types of datasets in immuneML:

- **RepertoireDatasets** should be used when making predictions per repertoire, such as predicting a disease state.

- **SequenceDatasets** should be used when predicting values for single immune receptor chains, such as antigen specificity.

- **ReceptorDatasets** are the paired variant of SequenceDatasets, and should be used to make a prediction for each receptor chain pair.

A broad range of different import formats can be specified, including AIRR, MiXCR, VDJdb, ImmunoSEQ (Adaptive Biotechnologies),
10xGenomics, OLGA and IGoR (for the extensive list, see :ref:`Datasets`). If you are using a custom format, or your preferred
format is not yet supported, any type of tabular file can also be imported using :ref:`Generic` import. When possible, using format-specific
importers is preferred over Generic import, as they require less options to be set and might take care of automatic reformatting
of certain fields.
Alternatively to importing data from files, it is also possible to generate datasets containing random immune receptor sequences on the fly,
see :ref:`How to generate a random receptor or repertoire dataset`.


Specifying data import from files
---------------------------------

Data import must be defined as a part of the YAML specification. First, we choose a name which will be used to refer to the dataset in the subsequent analyses:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        ... # here, format and input parameters will be specified

The name is defined by the user. It can consist of letters, numbers and underscores.

Under the dataset name key, the **format** of the data must be specified, as well as additional parameters under a key named **params**.
Under **format**, any of the formats listed under :ref:`Datasets` may be filled in. Under **params**, the parameter **path** is always
required when importing data from files. All the files must be stored in a single folder, and this folder must set through the
parameter **path**.

Here is an incomplete example specification using AIRR format:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          path: /path/to/data/
          ... # other import parameters will be specified here


Specifying params for repertoire dataset import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, it is assumed that a RepertoireDataset should be imported. In this case, the path to the **metadata_file**
must be specified. The metadata file is a .csv file which contains one repertoire (filename) per row, and the metadata
labels for that repertoire. These metadata labels can be used to train classifiers for.
For more details on structuring the metadata file, see :ref:`What should the metadata file look like?`.
Note that only the Repertoire files that are present in the metadata file will be imported.

Other parameters that are specific to the format may be specified under **params** as well, and are explained in more detail for each format
under :ref:`Datasets`.

A complete specification for importing a RepertoireDataset from AIRR format with default parameters may look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          # required parameters
          path: /path/to/data/
          metadata_file: /path/to/metadata.csv
          # is_repertoire is by default True, and may be omitted
          is_repertoire: True
          # Other parameters specific to AIRR data may be specified here


Specifying params for receptor or sequence dataset import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to import a Sequence- or ReceptorDataset, set the parameter **is_repertoire** to False, and set **paired** to either False (SequenceDataset)
or True (ReceptorDataset). For Sequence- and ReceptorDatasets, metadata labels must be specified directly as columns in the input files.
These metadata labels can be used to train classifiers for. For example, a column 'binding' can be added, which may have values 'true' and 'false'.
The metadata labels are specified through parameter **metadata_column_mapping**, which is a mapping from the names of the columns in
the file to the names that will be used internally in immuneML (for example: when specifying **labels** in the :ref:`TrainMLModel` instruction).
It is recommended that the immuneML-internal names contain only lowercase letters, numbers and underscores.

A complete specification for importing a SequenceDataset from AIRR format with default parameters may look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AIRR
        params:
          # required parameters
          path: /path/to/data/
          is_repertoire: False
          paired: False # must be true for ReceptorDatasets and False for SequenceDatasets
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

Some instructions (:ref:`Simulation`, :ref:`DatasetGeneration`, :ref:`SubSampling`) also explicitly export immuneML
datasets when selecting 'Pickle' as the export format.

These `.iml_dataset` files can later be imported easily and with few parameters, and importing from `.iml_dataset` is
also faster than importing from other data formats. A YAML specification could look like this:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset:
        format: Pickle
        params:
          path: /path/to/dataset.iml_dataset
          # specifying a metadata_file is optional, it will update the dataset using this new metadata.
          metadata_file: path/to/metadata.csv

