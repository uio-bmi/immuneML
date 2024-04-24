Definitions
===========

The different components used inside an immuneML analysis are called :code:`definitions`.
These analysis components are used inside :code:`instructions` to perform an analysis.

This page documents all possible definitions and their *parameters* in detail.
For general usage examples please check out the :ref:`Tutorials`.

Please use the menu on the right side of this page to navigate to the
documentation for the components of interest, or jump to one of the following
sections:

- :ref:`Datasets`
- :ref:`Encodings`
- :ref:`ML methods`
- :ref:`Reports`
- :ref:`Preprocessings`
- :ref:`Simulation`


Datasets
--------

Under the :code:`definitions/datasets` component, the user can specify how to import a dataset from files.
The file format determines which importer should be used, as listed below. See also: :ref:`How to import data into immuneML`.

For testing purposes, it is also possible to generate a random dataset instead of importing from files, using
:ref:`RandomReceptorDataset`, :ref:`RandomSequenceDataset` or :ref:`RandomRepertoireDataset` import types.
See also: :ref:`How to generate a dataset with random sequences`.


.. include:: ../../specs/definitions/datasets.rst

Encodings
---------

Under the :code:`definitions/encodings` component, the user can specify how to encode a given dataset.
An encoding is a numerical data representation, which may be used as input for a machine learning algorithm.


.. include:: ../../specs/definitions/encodings.rst

ML methods
-----------

Under the :code:`definitions/ml_methods` component, the user can specify different ML methods to use on a given encoded dataset.

From version 3, immuneML includes different types of ML methods:

- :ref:`**Classifiers**` which make predictions about data. See also :ref:`How to train and assess a receptor or repertoire-level ML classifier`.
- :ref:`**Generative models**` to generate new AIR sequences. **IN DEVELOPMENT**
- :ref:`**Dimensionality reduction methods**` **IN DEVELOPMENT**

When choosing which ML method(s) are most suitable for your use-case, please consider the following table:

.. csv-table:: ML methods properties
   :file: ../_static/files/ml_methods_properties.csv
   :header-rows: 1



.. include:: ../../specs/definitions/ml_methods.rst

Reports
---------

Under the :code:`definitions/ml_methods` component, the user can specify reports which visualise or summarise different properties
of the dataset or analysis.

Reports have been divided into different types. Different types of reports can be specified depending on which instruction is run. Click on the name of the report type to see more details.

- :ref:`**Data reports**` show some type of features or statistics about a given dataset.
- :ref:`**Encoding reports**` show some type of features or statistics about an encoded dataset, or may export relevant sequences or tables.
- :ref:`**ML model reports**` show some type of features or statistics about a single trained ML model (e.g., model coefficients).
- :ref:`**Train ML model reports**` plot general statistics or export data of multiple models simultaneously when running the :ref:`TrainMLModel` instruction (e.g., performance comparison between models).
- :ref:`**Multi dataset reports**` are special reports that can be specified when running immuneML with the :code:`MultiDatasetBenchmarkTool`. See Manuscript use case 1: :ref:`Robustness assessment` for an example.


.. include:: ../../specs/definitions/reports.rst


Preprocessings
---------------


Under the :code:`definitions/preprocessing_sequences` component, the user can specify different preprocessing steps to
apply to a dataset before performing an analysis. This is optional.

.. include:: ../../specs/definitions/preprocessings.rst

Simulation
----------

Under the :code:`definitions/simulation` component, the user can specify parameters necessary for simulating synthetic
immune signals into an AIRR dataset. See also :ref:`Dataset simulation with LIgO`.

.. include:: ../../specs/definitions/simulation.rst
