How to perform an exploratory data analysis
============================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: perform an exploratory data analysis
   :twitter:description: See tutorials on how to perform an exploratory data analysis in immuneML.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


To explore preprocessing, encodings and/or reports without running a machine learning
algorithm, the :ref:`ExploratoryAnalysis` instruction should be used. The components in the
definitions section are defined in the same manner as for all other instructions
(see: :ref:`How to specify an analysis with YAML`).
The instruction consists of a list of analyses to be performed. Each analysis should
contain at least a :code:`dataset` and a :code:`report`. Optionally, the analysis may also contain an
:code:`encoding` along with :code:`labels` if applicable.
In the example below, two analyses are done:

- :code:`my_analysis_1` runs report :code:`my_seq_lengths` directly on dataset :code:`my_dataset`

- :code:`my_analysis_2` first encodes :code:`my_dataset` using :code:`my_regex_matches` before running report :code:`my_matches`.

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # user-defined dataset name
        format: AIRR
        params:
          metadata_file: path/to/metadata.csv
          path: path/to/data/

    encodings:
      my_regex_matches:
        MatchedRegex:
          motif_filepath: path/to/regex_file.tsv

    reports:
      my_seq_lengths: SequenceLengthDistribution # reports without parameters
      my_matches: Matches

  instructions:
    my_instruction: # user-defined instruction name
      type: ExploratoryAnalysis
      analyses:
        my_analysis_1: # user-defined analysis name
          dataset: my_dataset
          report: my_seq_lengths
        my_analysis_2:
          dataset: my_dataset
          encoding: my_regex_matches
          report: my_matches

Where the file :download:`regex_file.tsv <../_static/files/regex_file.tsv>` must be a tab-separated file, which may contain the following contents:

====  ==========
id    TRB_regex
====  ==========
1     ACG
2     EDNA
3     DFWG
====  ==========

