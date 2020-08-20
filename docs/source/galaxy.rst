immuneML & Galaxy
=================

ImmuneML is integrated with Galaxy through a list of Galaxy tools. These tools include tools for repertoire and receptor classification for immunology
experts and CLI equivalent tools where the analysis is defined by YAML specification.

Tools for immunology experts:

1. :ref:`How to classify immune repertoires` - a tool with customized user interface for repertoire classification.

CLI equivalent tools based on YAML specification:

1. :ref:`How to make an immuneML dataset in Galaxy` - a tool that creates a Galaxy collection from a set of repertoire or receptor files and
corresponding metadata,

2. :ref:`How to run an analysis in Galaxy` - a tool that can perform any analysis immuneML supports using a Galaxy collection created in the ImmuneML
Dataset wrapper or raw files and a specification.


Galaxy tutorials:

.. toctree::
  :maxdepth: 1
  :caption: Galaxy tutorials:

  galaxy/how_to_make_an_immuneML_dataset_in_galaxy
  galaxy/how_to_run_an_analysis_in_galaxy
  galaxy/how_to_classify_immune_repertoires