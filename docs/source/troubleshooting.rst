Troubleshooting
===============

.. toctree::
   :maxdepth: 2

Installation issues
-------------------

When installing all requirements from requirements.txt, there is afterward an error with yaml package (No module named yaml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This issue might be helpful: https://github.com/yaml/pyyaml/issues/291. Try installing yaml manually with a specific version.

I get an error when installing PyTorch (could not find a version that satisfies the requirement torch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on the Python version and virtual environment, users may experience errors when installing PyTorch via pip.
The most common reason for this problem is that the Python version is too new to be compatible with the torch package.
Currently, the `torch package on pypi <https://pypi.org/project/torch/>`_ is only supported up to Python version 3.7.
We recommend trying to use Python version 3.7 or version 3.8 in a conda virtual environment.

If this does not resolve the problem, try installing torch manually using one of the commands described in `the PyTorch documentation <https://pytorch.org/get-started/previous-versions/>`_,
and afterwards try to install immuneML again.


Runtime issues
--------------

When running the TrainMLModel instruction multiple times, sometimes it fails saying that there is only one class in the data. Why does this happen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please check the number of examples used for machine learning (e.g. number of repertoires or receptors). If there are very few examples, and/or if classes
are not balanced, it is possible that just by chance, the data from only one class will be in the training set. If that happens, the classifiers
will not train and an error will be thrown. To fix this, try working with a larger dataset or check how TrainMLModel is specified.
If TrainMLModel does nested cross-validation, it might require a bit more data. To perform only cross-validation, under `selection` key, specify
that `split_strategy` is `random` and that `training_percentage` is `1` (to use all data from the inner loop for training). In this way, instead of having
multiple training/validation/test splits, there will be only training/test splits as specified under key `assessment` in TrainMLModel instruction.

I get an error when importing an immuneML dataset in Pickle format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
immuneML Pickle files are not guaranteed to be compatible between different immuneML (sub)versions.
Furthermore, the pickle protocol in Python 3.8 is 5, whereas in Python 3.7 it is 4 (immuneML always uses the highest protocol).

In the folder where your Pickle dataset was exported, you can find a file named info.txt, which shows the
immuneML version, Python version and pickle protocol that were used to export the given dataset.
If you install the same Python and immuneML version as displayed in this file, you should be able to import the Pickle dataset.

If you want to use your dataset with a different version of immuneML, try to load the dataset using the version that
was originally used to generate it and export the dataset to AIRR format. The exported AIRR dataset can subsequently
be imported in a different version of immuneML.

Converting a Pickle dataset to AIRR format can be done using the :ref:`DatasetExport` instruction.
An example YAML specification looks like this:

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset:
        format: Pickle
        params:
          path: path/to/dataset.iml_dataset # change the path to the location of the .iml_dataset file
  instructions:
    my_dataset_export_instruction:
      type: DatasetExport
      datasets:
      - my_dataset
    export_formats:
        - AIRR

When running DeepRC I get TypeError: can't concat str to bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This error occurs when h5py version 3 or higher is used. Try using version 2.10.0 or lower.
