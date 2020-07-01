How to import the data to immuneML
==================================

In immuneML, we need to import the data which we will use for the analysis. Importing the data is a part of the YAML specification and in this
tutorial it will be explained how to define it.

immuneML can import preprocessed immune receptor data (both single and paired chain) and preprocessed immune repertoire data. Immune receptor data
can be imported from VDJdb and IRIS formats, while repertoire can be imported from the data coming from Adaptive Biotechnologies, AIRR, MiXCR, Olga
(for extensive list, see :ref:`Datasets` under :ref:`Specification`). Importing a dataset in immuneML means that the data will be converted to an
optimized binary format which will speed up the analysis. All the data to be imported should be in the single folder which will be listed in the
specification afterwards.

The imported data creates a dataset which can be used in the analysis. The dataset and importing should be specified under definitions/datasets keys
in the YAML specification. First we define a name we will use for the dataset in the given analysis:

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the specification
        …. # here we will put the parameters on how to import the data

The name is defined by the user. It can consist of letters, numbers and underscore sign.

In addition to the name, we need to specify the format of the dataset (e.g. the data is in the AIRR format, or it should be imported from MiXCR or
Adaptive), and parameters that will define how to import the data. Here we will give an example of importing data from AdaptiveBiotech format
(a smaller version of the data available from Emerson et al. 2017 study, which is located under ImmuneML GitHub repository under directory datasets,
as shown below).

.. indent with spaces
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # this is the name of the dataset we will use in the specification
        format: AdaptiveBiotech # format of the dataset
        params:
          path: ImmuneML/datasets/cmv_emerson_2017/repertoires/
          metadata_file: ImmuneML/datasets/cmv_emerson_2017/cmv_emerson_2017_metadata.csv
          result_path: imported_dataset/ # where to store the imported data
          import_productive: True # should we import productive sequences
          import_out_of_frame: False # do not import sequences which are out of frame

The parameters are specific to the format. Please see the documentation under :ref:`Datasets` for all formats, parameters and valid values.