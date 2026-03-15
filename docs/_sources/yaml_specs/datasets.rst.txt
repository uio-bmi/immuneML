Dataset parameters
=======================


Under the :code:`definitions/datasets` component, the user can specify how to import a dataset from files.
The file format determines which importer should be used, as listed below. See also: :ref:`How to import data into immuneML`.

For testing purposes, it is also possible to generate a random dataset instead of importing from files, using
:ref:`RandomReceptorDataset`, :ref:`RandomSequenceDataset` or :ref:`RandomRepertoireDataset` import types.
See also: :ref:`How to generate a dataset with random sequences`.

.. note:: To reduce storage requirements with AIRR datasets

  In the standard workflow, the dataset is imported, preprocessed and stored under the results folder as specified by
  the user. If the dataset is already in the AIRR format and no preprocessing is needed, to avoid using too much
  storage, under datasets/dataset/params, it is possible to specify `result_path` parameter to be the same as the
  `path` from which the dataset is imported. In this case, the dataset will not be copied to the results folder,
  but used directly from the original location.


.. include:: ../../specs/definitions/specs_datasets.rst
