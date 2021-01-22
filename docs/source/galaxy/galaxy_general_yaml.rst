How to run any AIRR ML analysis in Galaxy
=========================================

To be able to run any possible YAML-based immuneML analysis in Galaxy, the tool 'Run immuneML with YAML specification' should be used.
It is typically recommended to use the analysis-specific Galaxy tools for :ref:`creating datasets <How to make an immuneML dataset in Galaxy>`,
:ref:`simulating synthetic data <How to simulate an AIRR dataset in Galaxy>`,
:ref:`implanting synthetic immune signals <How to simulate immune events into an existing AIRR dataset in Galaxy>` or
:ref:`training <How to train ML models in Galaxy>` and :ref:`applying <How to apply previously trained ML models to a new AIRR dataset in Galaxy>` ML models instead of this tool.
These other tools are able to export the relevant output files to Galaxy history elements.

However, when you want to run the :ref:`ExploratoryAnalysis` instruction, or other analyses that do not have a corresponding Galaxy tool, this generic tool can be used.


Creating the YAML specification
---------------------------------------------

This Galaxy tool takes as input an immuneML dataset from the Galaxy history, optional additional files, and a YAML specification file.
To see the details on how to write the YAML specification, see :ref:`How to specify an analysis with YAML`.

When writing an analysis specification for Galaxy, it can be assumed that all selected files are present in the current working directory. A path
to an additional file thus consists only of the filename.

The following YAML specification shows an example of how to run the ExploratoryAnalysis instruction inside Galaxy:


.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      dataset: # user-defined dataset name
        format: Pickle # the default format used by the 'Create dataset' galaxy tool is Pickle
        params:
          path: dataset.iml_dataset # specify the dataset name, the default name used by
                                    # the 'Create dataset' galaxy tool is dataset.iml_dataset
    encodings:
      my_regex_matches:
        MatchedRegex:
          motif_filepath: regex_file.tsv # this file must be selected from the galaxy history as an 'additional file'

    reports:
      my_seq_lengths: SequenceLengthDistribution # reports without parameters
      my_matches: Matches

  instructions:
    my_instruction: # user-defined instruction name
      type: ExploratoryAnalysis
      analyses:
        my_analysis_1: # user-defined analysis name
          dataset: dataset
          report: my_seq_lengths
        my_analysis_2:
          dataset: dataset
          encoding: my_regex_matches
          report: my_matches

Where the file regex_file.tsv must be a tab-separated file, which may contain the following lines:

====  ==========
id    TRB_regex
====  ==========
1     ACG
2     EDNA
3     DFWG
====  ==========

