FAQ
===

.. toctree::
   :maxdepth: 2

What should the metadata file look like?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The metadata file is defined for repertoire datasets only (the metadata information for receptor and sequence datasets are defined in the same file
as the sequences and receptors).

In case of repertoire datasets, each repertoire is represented by one file in the given format (e.g., AIRR/MiXCR/Adaptive).
For all repertoires in one dataset, a single metadata file should be defined. The metadata file should be a .csv file and have the following columns:

.. image:: _static/images/metadata.png

The columns filename and subject_id are mandatory. Other columns may be defined by the user. There are no restrictions on what those should include,
but often other columns will have information on HLA, age, sex, and specific diseases. Any of these columns may be used as a prediction target in
the downstream analysis. The prediction target is specified by the name of the column in the YAML specification as the value of the label parameter.
For more information on YAML specification, see :ref:`YAML specification`.

When installing all requirements from requirements.txt, there is afterward an error with yaml package (No module named yaml)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This issue might be helpful: https://github.com/yaml/pyyaml/issues/291. Try installing yaml manually with a specific version.

I get an error when installing PyTorch (could not find a version that satisfies the requirement torch==1.5.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on the Python version and virtual environment, users may experience errors when installing PyTorch via pip.
The most common reason for this problem is if Python 3.9. We recommend trying to use Python version 3.7 or 3.8 in a conda
virtual environment.
If this does not resolve the problem, try installing torch v1.5.1 manually using one of the commands described in `the PyTorch documentation <https://pytorch.org/get-started/previous-versions/>`_,
and afterwards try to install immuneML again.


There is an issue with the type of entry when specifying a list of inputs, why does this happen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please check that the YAML is in valid format. To list different inputs (e.g. a list of reports under assessment/reports/encoding in
TrainMLModel instruction), the correct YAML syntax includes a space between - and the list item.

When running the TrainMLModel instruction multiple times, sometimes it fails saying that there is only one class in the data. Why does this happen?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please check the number of examples used for machine learning (e.g. number of repertoires). If there are very few examples, and/or if classes
are not balanced, it is possible that just by chance, the data from only one class will be in the training set. If that happens, the classifiers
will not train and an error will be thrown. To fix this, try working with a larger dataset or check how TrainMLModel is specified.
If TrainMLModel does nested cross-validation, it might require a bit more data. To perform only cross-validation, under `selection` key, specify
that `split_strategy` is `random` and that `training_percentage` is `1` (to use all data from the inner loop for training). In this way, instead of having
multiple training/validation/test splits, there will be only training/test splits as specified under key `assessment` in TrainMLModel instruction.

When running DeepRC I get TypeError: can't concat str to bytes, how can I solve this?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This error occurs when h5py version 3 or higher is used. Try using version 2.10.0 or lower.
