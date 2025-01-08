Report parameters
=======================

Under the :code:`definitions/reports` component, the user can specify reports which visualise or summarise different properties
of the dataset or analysis.

Reports have been divided into different types. Different types of reports can be specified depending on which instruction is run. Click on the name of the report type to see more details.

- :ref:`**Data reports**` show some type of features or statistics about a given dataset.
- :ref:`**Encoding reports**` show some type of features or statistics about an encoded dataset, or may export relevant sequences or tables.
- :ref:`**ML model reports**` show some type of features or statistics about a single trained ML model (e.g., model coefficients).
- :ref:`**Train ML model reports**` plot general statistics or export data of multiple models simultaneously when running the :ref:`TrainMLModel` instruction (e.g., performance comparison between models).
- :ref:`**Multi dataset reports**` are special reports that can be specified when running immuneML with the :code:`MultiDatasetBenchmarkTool`. See Manuscript use case 1: :ref:`Robustness assessment` for an example.


.. include:: ../../specs/definitions/reports.rst
