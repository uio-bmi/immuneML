immuneML & Galaxy
=================

All of immuneMLs functionalities are also available through a Galaxy web interface as a collection of Galaxy tools. We provide a YAML-based Galaxy
tool that is equivalent to the CLI (command-line interface), as well as repertoire and receptor-level classification tools with an intuitive
graphical user interface aimed at immunology experts without a machine learning background.

To get started, you will need to add your dataset to Galaxy, which is explained in this tutorial:

- :ref:`How to make an immuneML dataset in Galaxy` - a tool that creates a Galaxy collection from a set of repertoire or receptor files and
corresponding metadata

Subsequently, immunology experts without machine learning background can follow these instructions:

- :ref:`Classify immune repertoires` (Galaxy tool) - a tool with an easily interpretable user interface for repertoire classification (e.g., immune status prediction).

- Classify immune receptors (Galaxy tool) - coming soon

Alternatively, CLI equivalent tools based on the YAML specification can be run using the following instructions

-  :ref:`How to run an analysis in Galaxy` - a tool that can perform any analysis immuneML supports using a Galaxy collection created in the
‘Create dataset’ Galaxy tool or raw files and a YAML specification.


Galaxy tutorials:

.. toctree::
  :maxdepth: 1
  :caption: Galaxy tutorials:

  galaxy/how_to_make_an_immuneML_dataset_in_galaxy
  galaxy/how_to_run_an_analysis_in_galaxy
  galaxy/how_to_classify_immune_repertoires
  galaxy/how_to_classify_immune_receptors