How to perform an exploratory data analysis
============================================

To explore preprocessing, encodings and/or reports without running a machine learning
algorithm, the ExploratoryAnalysis instruction should be used. The components in the
definitions section are defined in the same manner as for all other instructions
(see: :ref:`How to specify an analysis with YAML`).
The instruction consists of a list of analyses to be performed. Each analysis should
contain at least a dataset and a report. Optionally, the analysis may also contain an
encoding along with the labels. Encoding reports can be run only if encoding and labels
are defined. In the example below, my_analysis_1 runs report my_seq_lengths directly on dataset my_dataset,
whereas in my_analysis_2 dataset my_dataset is encoded first using my_regex_matches before running report my_matches.

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      my_dataset: # user-defined dataset name
        format: AIRR
        params:
          metadata_file: /path/to/metadata.csv
          path: /path/to/data/

    encodings:
      my_regex_matches:
        MatchedRegex:
          motif_filepath: /path/to/file.tsv

    reports:
      my_seq_lengths: SequenceLengthDistribution # a report with default parameters
      my_matches: Matches

  instructions:
    instruction_1:
      type: ExploratoryAnalysis
      analyses:
        my_analysis_1: # this user-defined name of the analysis is later used as a folder name in results
          dataset: my_dataset
          report: my_seq_lengths
        my_analysis_2:
          dataset: my_dataset
          encoding: my_regex_matches
          report: my_matches
          labels:
              - disease

Where the file regex_file.tsv must be a tab-separated file, which may contain the following lines:

====  ==========
id    TRB_regex
====  ==========
1     ACG
2     EDNA
3     DFWG
====  ==========

