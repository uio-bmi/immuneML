How to import data into immuneML
==================================

In immuneML, the first step of an analysis is to import the AIRR data which we will use. Importing the data is a part of the YAML specification and
in this tutorial it will be explained how to define it.

immuneML can import preprocessed immune *receptor* data (both single and paired chain) and preprocessed immune *repertoire* data. Immune receptor data
can be imported from VDJdb or 10xGenomics, while repertoire can be imported from data coming from Adaptive Biotechnologies, AIRR, MiXCR, OLGA, IGoR
(for the extensive list, see :ref:`Datasets` under :ref:`YAML specification`). Importing a dataset in immuneML means that the data will be converted to an
optimized binary format, which will speed up the analysis. All the data to be imported should be stored in a single folder, which must be provided in the
YAML specification.

The imported data creates a dataset which can be used in the analysis. The dataset and import settings (e.g., format) should be specified under the
definitions/datasets keys in the YAML specification. First, we choose a name which will be used to refer to the dataset in the subsequent analyses:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        …. # here we will put the parameters on how to import the data

The name is defined by the user. It can consist of letters, numbers and underscores.

In addition to the name, we need to specify the format of the dataset (e.g., the data is in the AIRR format, or it should be imported from MiXCR or
Adaptive), and parameters that will define how to import the data. Here, we will give an example of importing data from AdaptiveBiotech format
(a smaller version of the data available from Emerson et al. 2017 study, which is located at the immuneML GitHub repository under directory datasets,
as shown below).

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the YAML specification
        format: AdaptiveBiotech # format of the dataset
        params:
          path: immuneML/datasets/cmv_emerson_2017/repertoires/
          metadata_file: immuneML/datasets/cmv_emerson_2017/cmv_emerson_2017_metadata.csv
          result_path: imported_dataset/ # where to store the imported data
          import_productive: True # should we import productive sequences
          import_out_of_frame: False # do not import sequences which are out of frame

The parameters are specific to the format. Please see the documentation under :ref:`Datasets` for all formats, parameters and valid values.