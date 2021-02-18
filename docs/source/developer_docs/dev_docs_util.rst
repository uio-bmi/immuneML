.. note::

  When adding new features to immuneML, some utility classes are already available. For instance, to construct a path, you can use :code:`PathBuilder.build()` function.
  If you need to validate some parameters when constructing an object in :code:`build_object()` functions, for example, you can use :code:`ParameterValidator` class.
  For the full list of such classes, see the `immuneML.util` package.

.. note::

  To access the information about the datasets, see the corresponding dataset class: :py:obj:`immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset`
  when working on the repertoire level, and :py:obj:`immuneML.data_model.dataset.SequenceDataset.SequenceDataset` (single chain) and
  :py:obj:`immuneML.data_model.dataset.ReceptorDataset.ReceptorDataset` (paired chain) when working on the sequence level. Implementation details for
  these two receptor dataset classes are available in :py:obj:`immuneML.data_model.dataset.ElementDataset.ElementDataset`.

  Useful function in the dataset classes include getting the metadata information from the :py:obj:`immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset`,
  using :py:obj:`immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset.get_metadata` function, obtaining the number of examples in the
  dataset, checking possible labels or making subsets.
