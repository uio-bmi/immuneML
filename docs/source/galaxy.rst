immuneML & Galaxy
=================

All of immuneMLs functionalities are also available through a Galaxy web interface as a collection of Galaxy tools. We provide a YAML-based Galaxy
tool that is equivalent to the CLI (command-line interface), as well as repertoire and receptor-level classification tools with an intuitive
graphical user interface aimed at immunology experts without a machine learning background.

To get started, you will need to add your dataset to Galaxy, which is explained in this tutorial:

- :ref:`How to make an immuneML dataset in Galaxy` - a tool that creates a Galaxy collection from a set of repertoire or receptor files and corresponding metadata.

If you do not want to use experimental data and just want to try something out quickly, you can simulate an immune dataset:

- :ref:`Simulate an immune receptor or repertoire dataset` - a tool that creates a simple immune repertoire or receptor dataset for bechmarking or testing purposes.

Subsequently, immunology experts without machine learning background can follow these instructions:

- :ref:`Train immune repertoire classifiers` (Galaxy tool) - a tool with an easily interpretable user interface for repertoire classification (e.g., immune status prediction).

- :ref:`Train immune receptor classifiers` (Galaxy tool) - a tool with an easily interpretable user interface for antigen binding prediction

Alternatively, CLI equivalent tools based on the YAML specification can be run using the following instructions

-  :ref:`How to run an analysis in Galaxy` - a tool that can perform any analysis immuneML supports using a Galaxy collection created in the ‘Create dataset’ Galaxy tool or raw files and a YAML specification.


.. toctree::
  :maxdepth: 1
  :caption: Galaxy tutorials:

  galaxy/how_to_make_an_immuneML_dataset_in_galaxy
  galaxy/how_to_simulate_immune_dataset
  galaxy/how_to_run_an_analysis_in_galaxy
  galaxy/how_to_classify_immune_repertoires
  galaxy/how_to_classify_immune_receptors