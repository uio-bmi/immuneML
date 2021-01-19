Developer documentation
=======================

  immuneML has a modular architecture and can be easily extended with new functionalities. Specifically, see the tutorials on:

  - :ref:`How to add a new machine learning method` - new ML methods can subsequently be trained and used for classification of immune receptors or repertoires,
  - :ref:`How to add a new encoding` - encodings are used to represent (encode) the immune receptor data, to use as input for a machine learning method or a report to provide additional insights,
  - :ref:`How to add a new report` - reports can be used to examine the inner mechanisms of machine learning methods, encoded receptor data or to perform an exploratory analysis.


.. toctree::
  :maxdepth: 1
  :caption: Developer tutorials:

  developer_docs/how_to_add_new_ML_method.rst
  developer_docs/how_to_add_new_encoding.rst
  developer_docs/how_to_add_new_report.rst

.. toctree::
  :maxdepth: 4
  :caption: List of all classes:

  modules