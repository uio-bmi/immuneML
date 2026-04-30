Instruction parameters
=======================

The different workflows that can be executed by immuneML are called :code:`instructions`.
Different instructions may require different analysis components (defined under :code:`definitions`).

This page documents all instructions and their *parameters* in detail.
Tutorials for general usage of most instructions can be found under :ref:`Tutorials`.

Please use the menu on the right side of this page to navigate to the
documentation for the instructions of interest, or jump to one of the following
sections:

**Machine learning:**

- :ref:`TrainMLModel`: select and fit a classifier,
- :ref:`MLApplication`: apply fitted classifier to a new dataset,
- :ref:`TrainGenModel`: train (multiple) generative model(s),
- :ref:`ApplyGenModel`: apply a generative model to make new receptor sequences.

**Data simulation:**

- :ref:`LigoSim`: use LIgO tool to generate synthetic datasets with different immune events and signals,
- :ref:`FeasibilitySummary`: check the feasibility of the simulation with provided parameters.

**Data analysis, exploration and manipulation:**

- :ref:`ExploratoryAnalysis`: various data summaries for both raw and encoded data,
- :ref:`Clustering`: fit and compare multiple clustering settings,
- :ref:`ValidateClustering`: validate a clustering setting on a new dataset,
- :ref:`DatasetExport`: export a dataset,
- :ref:`Subsampling`: subsample a dataset and export as a new dataset,
- :ref:`SplitDataset`: split dataset into two (e.g., for discovery and validation datasets for clustering).


.. include:: ../../specs/instructions/instructions.rst
