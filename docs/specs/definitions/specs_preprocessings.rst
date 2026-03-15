
ChainRepertoireFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This filter has two options: it can remove repertoires from the dataset which have any chain other than the
specified one (e.g., keep  only TRB) or it can remove the sequences from the repertoire which do not have the
desired chain.

Since the filter may remove repertoires/sequences from the dataset (examples in machine learning setting), it
cannot be used with :ref:`TrainMLModel` instruction. If you want to filter out repertoires including a given chain,
see :ref:`DatasetExport` instruction with preprocessing.

**Dataset types:**

- RepertoireDataset

**Specification arguments:**

- keep_chains (list): Which chains should be kept, valid values are "TRA", "TRB", "IGH", "IGL", "IGK"

- remove_only_sequences (bool): Whether to remove only sequences with different chain than "keep_chain" (true) in
  case of repertoire datasets; default is false

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter:
                ChainRepertoireFilter:
                    keep_chains: [TRB]
                    remove_only_sequences: true



ClonesPerRepertoireFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Removes all repertoires from the RepertoireDataset, which contain fewer clonotypes than specified by the
lower_limit, or more clonotypes than specified by the upper_limit.
Note that this filter filters out repertoires, not individual sequences, and can thus only be applied to RepertoireDatasets.
When no lower or upper limit is specified, or the value -1 is specified, the limit is ignored.

Since the filter removes repertoires from the dataset (examples in machine learning setting), it cannot be used with :ref:`TrainMLModel`
instruction. If you want to use this filter, see :ref:`DatasetExport` instruction with preprocessing.

**Specification arguments:**

- lower_limit (int): The minimal inclusive lower limit for the number of clonotypes allowed in a repertoire.

- upper_limit (int): The maximal inclusive upper limit for the number of clonotypes allowed in a repertoire.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter:
                ClonesPerRepertoireFilter:
                    lower_limit: 100
                    upper_limit: 100000



CountPerSequenceFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Removes all sequences from a Repertoire when they have a count below low_count_limit, or sequences with no count
value if remove_without_counts is True. This filter can be applied to Repertoires and RepertoireDatasets.

**Specification arguments:**

- low_count_limit (int): The inclusive minimal count value in order to retain a given sequence.

- remove_without_count (bool): Whether the sequences without a reported count value should be removed.

- remove_empty_repertoires (bool): Whether repertoires without sequences should be removed.
  Only has an effect when remove_without_count is also set to True. If this is true, this preprocessing cannot be used with :ref:`TrainMLModel`
  instruction, but only with :ref:`DatasetExport` instruction instead.

- batch_size (int): number of repertoires that can be loaded at the same time (only affects the speed when applying this filter on a RepertoireDataset)


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter:
                CountPerSequenceFilter:
                    remove_without_count: True
                    remove_empty_repertoires: True
                    low_count_limit: 3
                    batch_size: 4



DuplicateSequenceFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Collapses duplicate nucleotide or amino acid sequences within each repertoire in the given RepertoireDataset or within a SequenceDataset.
This filter can be applied to Repertoires, RepertoireDatasets, and SequenceDatasets.

Sequences are considered duplicates if the following fields are identical:

- amino acid or nucleotide sequence (whichever is specified)

- v and j genes (note that the full field including subgroup + gene is used for matching, i.e. V1 and V1-1 are not considered duplicates)

- chain

- region type

For all other fields (the non-specified sequence type, custom lists, sequence identifier) only the first occurring
value is kept.

Note that this means the count value of a sequence with a given sequence identifier might not be the same as before
removing duplicates, unless count_agg = FIRST is used.

**Specification arguments:**

- filter_sequence_type (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Whether the sequences should be collapsed on the nucleotide or amino acid level. Valid values are: ['amino_acid', 'nucleotide'].

- region_type (str): which part of the sequence to examine, by default, this is IMGT_CDR3

- count_agg (:py:obj:`~immuneML.preprocessing.filters.CountAggregationFunction.CountAggregationFunction`): determines how the sequence counts of duplicate sequences are aggregated. Valid values are: ['sum', 'max', 'min', 'mean', 'first', 'last'].


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter:
                DuplicateSequenceFilter:
                    # required parameters:
                    filter_sequence_type: AMINO_ACID
                    # optional parameters (if not specified the values bellow will be used):
                    batch_size: 4
                    count_agg: SUM
                    region_type: IMGT_CDR3



MetadataFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Removes examples from a dataset based on the examples' metadata. It works for any dataset type. Note that
for repertoire datasets, this means that repertoires will be filtered out, and for sequences datasets - sequences.

Since this filter changes the number of examples, it cannot be used with
:ref:`TrainMLModel` instruction. Use with DatasetExport instruction instead.

**Specification arguments:**

- criteria (dict): a nested dictionary that specifies the criteria for keeping the dataset examples based on the
  column values; it contains the type of evaluation, name of the column, and additional parameters depending on
  evaluation; alternatively, it can contain a combination of multiple (evaluation, column, parameters) groups;
  evaluation_types: IN, NOT_IN, NOT_NA, GREATER_THAN, LESS_THAN, TOP_N, RANDOM_N; for IN, NOT_IN the parameter name
  is 'values', for GREATER_THAN, LESS_THAN the parameter name is 'threshold' and for TOP_N, RANDOM_N the parameter
  name is 'number'; supported boolean combinations of groups are AND and OR with (evaluation, column, parameter)
  groups specified under 'operands' key; see the YAML below for example.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter:
                # Example filter that keeps e.g., repertoires with values greater than 1 in the "my_column_name"
                # column of the metadata_file
                MetadataFilter:
                    type: GREATER_THAN
                    column: my_column_name
                    threshold: 1
        my_second_preprocessing:
            - my_filter2: # only examples which in column "label" have values 'label_val1' or 'label_val2' are kept
                MetadataFilter:
                    type: IN
                    values: [label_val1, label_val2]
                    column: label
        my_third_preprocessing_example:
            - my_combined_filter:
                MetadataFilter:
                # keeps examples with that have label_val1 or label_val2 in the column label and
                # that at the same time have a value larger than 1.3 in another_metadata_column
                    type: AND
                    operands:
                    - type: IN
                      values: [label_val1, label_val2]
                      column: label
                    - type: GREATER_THAN
                      column: another_metadata_column
                      threshold: 1.3



ReferenceSequenceAnnotator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Annotates each sequence in each repertoire if it matches any of the reference sequences provided as input parameter. This report uses CompAIRR internally. To match CDR3 sequences (and not JUNCTION), CompAIRR v1.10 or later is needed.

**Specification arguments:**

- reference_sequences (dict): A dictionary describing the reference dataset file. Import should be specified the same way as regular dataset import. It is only allowed to import a receptor dataset here (i.e., is_repertoire is False and paired is True by default, and these are not allowed to be changed).

- max_edit_distance (int): The maximum edit distance between a target sequence (from the repertoire) and the reference sequence.

- compairr_path (str): optional path to the CompAIRR executable. If not given, it is assumed that CompAIRR has been installed such that it can be called directly on the command line with the command 'compairr', or that it is located at /usr/local/bin/compairr.

- threads (int): how many threads to be used by CompAIRR for sequence matching

- ignore_genes (bool): Whether to ignore V and J gene information. If False, the V and J genes between two receptor chains have to match. If True, gene information is ignored. By default, ignore_genes is False.

- output_column_name (str): in case there are multiple annotations, it is possible here to define the name of the column in the output repertoire files for this specific annotation

- repertoire_batch_size (int): how many repertoires to process simultaneously; depending on the repertoire size, this parameter might be use to limit the memory usage

- region_type (str): which region type to check, default is IMGT_CDR3


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - step1:
                ReferenceSequenceAnnotator:
                    reference_sequences:
                        format: VDJDB
                        params:
                            path: path/to/file.csv
                    compairr_path: optional/path/to/compairr
                    ignore_genes: False
                    max_edit_distance: 0
                    output_column_name: matched
                    threads: 4
                    repertoire_batch_size: 5
                    region_type: IMGT_CDR3



SequenceLengthFilter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Removes sequences with length out of the predefined range.

**Supported dataset types:**

- SequenceDataset

- ReceptorDataset

- RepertoireDataset


**Specification arguments:**

- sequence_type (:py:obj:`~immuneML.environment.SequenceType.SequenceType`): Whether the sequences should be filtered on the nucleotide or amino acid level. Valid options are defined by the SequenceType enum.

- min_len (int): minimum length of the sequence (sequences shorter than min_len will be removed); to not use min_len, set it to -1

- max_len (int): maximum length of the sequence (sequences longer than max_len will be removed); to not use max_len, set it to -1

- region_type (str): which part of the sequence to examine, by default, this is IMGT_CDR3

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter:
                SequenceLengthFilter:
                    sequence_type: AMINO_ACID
                    min_len: 3 # -> remove all sequences shorter than 3
                    max_len: -1 # -> no upper bound on the sequence length

    

SubjectRepertoireCollector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Merges all the Repertoires in a RepertoireDataset that have the same 'subject_id' specified in the metadata. The result
is a RepertoireDataset with one Repertoire per subject. This preprocessing cannot be used in combination with :ref:`TrainMLModel`
instruction because it can change the number of examples. To combine the repertoires in this way, use this preprocessing
with :ref:`DatasetExport` instruction.


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    preprocessing_sequences:
        my_preprocessing:
            - my_filter: SubjectRepertoireCollector


