immuneML & Galaxy
=================

All of immuneMLs functionalities are also available through a Galaxy web interface as a collection of Galaxy tools. We provide a YAML-based Galaxy
tool that is equivalent to the CLI (command-line interface), as well as repertoire and receptor-level classification tools with an intuitive
graphical user interface aimed at immunology experts without a machine learning background.

To get started, you will need to add your dataset to Galaxy, which is explained in this tutorial:

- :ref:`How to make an immuneML dataset in Galaxy` - how to use the 'Create dataset' tool to add an immuneML Galaxy dataset to the Galaxy history

Remote datasets may be fetched from VDJdb or the iReceptor Plus Gateway, see:

- :ref:`How to import remote datasets into immuneML` - how to work with data from remote sources, and import this data as an immuneML Galaxy dataset.

If you do not want to use experimental data and just want to try something out quickly, you can simulate an immune dataset:

- :ref:`Simulate an immune receptor or repertoire dataset` - create a simple immune repertoire or receptor dataset for testing or benchmarking purposes.

Subsequently, immunology experts without machine learning background can follow these instructions:

- :ref:`Train immune repertoire classifiers` (Galaxy tool) - a tool with an easily interpretable user interface for repertoire classification (e.g., immune status prediction).

- :ref:`Train immune receptor classifiers` (Galaxy tool) - a tool with an easily interpretable user interface for receptor classification (e.g., antigen binding prediction).

Alternatively, CLI equivalent tools based on the YAML specification can be run using the following instructions

- :ref:`How to run an analysis in Galaxy` - a tool that can perform any analysis immuneML supports using a Galaxy collection created in the ‘Create dataset’ Galaxy tool or raw files and a YAML specification.


.. toctree::
  :maxdepth: 1
  :caption: Galaxy tutorials:

  galaxy/how_to_make_an_immuneML_dataset_in_galaxy
  galaxy/how_to_import_remote_data.rst
  galaxy/how_to_simulate_immune_dataset
  galaxy/how_to_run_an_analysis_in_galaxy
  galaxy/how_to_classify_immune_repertoires
  galaxy/how_to_classify_immune_receptors