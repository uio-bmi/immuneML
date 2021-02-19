immuneML data model
=====================

immuneML data model supports three types of datasets that can be used for analyses:

#. Repertoire dataset - one example in the dataset is one repertoire typically coming from one subject
#. Receptor dataset - one example is one receptor with both chains set
#. Sequence dataset - one example is one receptor sequence with single chain information.

These types of datasets are implemented using the corresponding classes: :py:obj:`immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset`,
:py:obj:`immuneML.data_model.dataset.ReceptorDataset.ReceptorDataset` and :py:obj:`immuneML.data_model.dataset.SequenceDataset.SequenceDataset`.

Useful function in the dataset classes include getting the metadata information from the :py:obj:`immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset`,
using :py:obj:`immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset.get_metadata` function, obtaining the number of examples in the
dataset, checking possible labels or making subsets.

The UML diagram showing these classes and the underlying dependencies is shown below.

.. figure:: ../_static/images/dev_docs/data_model_architecture.png
  :width: 70%
  :alt: UML diagram of the immuneML's data model

  UML diagram showing the immuneML data model, where white classes are abstract defining the interface only, while green are concrete and used throughout the codebase.

Implementation details for :code:`ReceptorDataset` and :code:`SequenceDataset` are available in :py:obj:`immuneML.data_model.dataset.ElementDataset.ElementDataset`.