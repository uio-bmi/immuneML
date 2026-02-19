
AIRR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data in AIRR format into a Repertoire-, Sequence- or ReceptorDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
immune receptors respectively, like antigen specificity.

The AIRR .tsv format is explained here: https://docs.airr-community.org/en/stable/datarep/format.html
And the AIRR rearrangement schema can be found here: https://docs.airr-community.org/en/stable/datarep/rearrangements.html

When importing a ReceptorDataset, the AIRR field cell_id is used to determine the chain pairs.

**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with AIRR files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the AIRR files included under the column 'filename' are imported into the RepertoireDataset. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on the identifier given in the AIRR column named 'cell_id'.

- receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values are `TRA_TRB`, `TRG_TRD`, `IGH_IGL`, `IGH_IGK`. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

- import_productive (bool): Whether productive sequences (with value 'T' in column productive) should be included in the imported sequences. By default, import_productive is True.

- import_unknown_productivity (bool): Whether sequences with unknown productivity (missing value in column productive) should be included in the imported sequences. By default, import_unknown_productivity is True.

- import_with_stop_codon (bool): Whether sequences with stop codons (with value 'T' in column stop_codon) should be included in the imported sequences. This only applies if column stop_codon is present. By default, import_with_stop_codon is False.

- import_out_of_frame (bool): Whether out of frame sequences (with value 'F' in column vj_in_frame) should be included in the imported sequences. This only applies if column vj_in_frame is present. By default, import_out_of_frame is False.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- column_mapping (dict): A mapping from AIRR column names to immuneML's internal data representation. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the AIRR file, or using alternative column names).

    .. indent with spaces
    .. code-block:: yaml

        additional_column_in_the_file: column_name_to_be_used_in_analysis

- separator (str): Column separator, for AIRR this is by default "\t".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_airr_dataset:
                format: AIRR
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    import_productive: True # whether to include productive sequences in the dataset
                    import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                    import_out_of_frame: False # whether to include out of frame sequences in the dataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even if the `sequences` column is empty (provided that other fields are as specified here)
                    import_empty_aa_sequences: False # remove all sequences with empty column
                    # Optional fields with AIRR-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    region_type: IMGT_CDR3 # what part of the sequence check for import



Generic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data from any tabular file into a Repertoire-, Sequence- or ReceptorDataset. RepertoireDatasets should be
used when making predictions per repertoire, such as predicting a disease state. SequenceDatasets or ReceptorDatasets
should be used when predicting values for unpaired (single-chain) and paired immune receptors respectively,
like antigen specificity.

This importer works similarly to other importers, but has no predefined default values for which fields are imported,
and can therefore be tailored to import data from various different tabular files with headers.

For ReceptorDatasets: this importer assumes the two receptor sequences appear on different lines in the file, and can
be paired together by a common sequence identifier.


**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on a common identifier. This identifier should be mapped to the immuneML field 'sequence_identifiers' using the column_mapping.

- receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values are `TRA_TRB`, `TRG_TRD`, `IGH_IGL`, `IGH_IGK`.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means immuneML assumes the IMGT junction (including leading C and trailing Y/F amino acids) is used in the input file, and the first and last amino acids will be removed from the sequences to retrieve the IMGT CDR3 sequence. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- column_mapping (dict): Required for all datasets. A mapping where the keys are the column names in the input file, and the values correspond to the names in the AIRR format. Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. A column mapping can look for example like this:

    .. indent with spaces
    .. code-block:: yaml

        file_column_amino_acids: cdr3_aa
        file_column_v_genes: v_call
        file_column_j_genes: j_call
        file_column_frequencies: duplicate_count

- column_mapping_synonyms (dict): This is a column mapping that can be used if a column could have alternative names. The formatting is the same as column_mapping. If some columns specified in column_mapping are not found in the file, the columns specified in column_mapping_synonyms are instead attempted to be loaded. For Generic import, there is no default column_mapping_synonyms.

- columns_to_load (list): Optional; specifies which columns to load from the input file. This may be useful if the input files contain many unused columns. If no value is specified, all columns are loaded.

- separator (str): Required parameter. Column separator, for example "\t" or ",". The default value is "\t"


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_generic_dataset:
                format: Generic
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                    receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                    separator: "\t" # column separator
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have amino acid sequence set
                    region_type: IMGT_CDR3 # which column to check for illegal characters/empty strings etc
                    column_mapping: # column mapping file: immuneML/AIRR column names
                        file_column_amino_acids: junction_aa
                        file_column_v_genes: v_call
                        file_column_j_genes: j_call
                        file_column_frequencies: duplicate_count
                        file_column_antigen_specificity: antigen_specificity
                    columns_to_load:  # which subset of columns to load from the file
                        - file_column_amino_acids
                        - file_column_v_genes
                        - file_column_j_genes
                        - file_column_frequencies
                        - file_column_antigen_specificity



IGoR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data generated by `IGoR <https://github.com/qmarcou/IGoR>`_ simulations into a Repertoire-, or SequenceDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
antigen specificity.

Note that you should run IGoR with the --CDR3 option specified, this tool imports the generated CDR3 files.
Sequences with missing anchors are not imported, meaning only sequences with value '1' in the anchors_found column are imported.
Nucleotide sequences are automatically translated to amino acid sequences.

Reference: Quentin Marcou, Thierry Mora, Aleksandra M. Walczak
‘High-throughput immune repertoire analysis with IGoR’. Nature Communications, (2018)
`doi.org/10.1038/s41467-018-02832-w <https://doi.org/10.1038/s41467-018-02832-w>`_.

**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with IGoR files to import. For Sequence- or
  ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one
  or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset.
  By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
  This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in
  instructions. Only the IGoR files included under the column 'filename' are imported into the RepertoireDataset.
  For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column
  names of labels to import. These labels can be used as prediction target. When label_columns are not set, label
  names are attempted to be discovered automatically (any column name which is not used in the column_mapping).
  For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- import_with_stop_codon (bool): Whether sequences with stop codons should be included in the imported sequences.
  By default, import_with_stop_codon is False.

- import_out_of_frame (bool): Whether out of frame sequences (with value '0' in column is_inframe) should be
  included in the imported sequences. By default, import_out_of_frame is False.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters
  that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to
  false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only
  entries with illegal characters in the amino acid sequence are removed). By default, import_illegal_characters
  is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True
  or False. By default, import_empty_nt_sequences is set to True.

- region_type (str): Which part of the sequence to check when importing. By default, this value is set to IMGT_CDR3.
  This means the first and last amino acids are removed from the CDR3 sequence, as IGoR uses the IMGT junction.
  Specifying any other value will result in importing the sequences as they are. Valid values for region_type are
  the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

- column_mapping (dict): A mapping from IGoR column names to immuneML's internal data representation. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the IGoR file, or using alternative column names). Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. For IGoR, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

        nt_CDR3: cdr3
        seq_index: sequence_id

- separator (str): Column separator, for IGoR this is by default ",".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_igor_dataset:
                format: IGoR
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                    import_out_of_frame: False # whether to include out of frame sequences in the dataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    # Optional fields with IGoR-specific defaults, only change when different behavior is required:
                    separator: "," # column separator
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    column_mapping: # column mapping IGoR: immuneML
                        nt_CDR3: cdr3
                        seq_index: sequence_id
                        igor_column_name1: metadata_label1
                        igor_column_name2: metadata_label2



IReceptor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports AIRR datasets retrieved through the `iReceptor Gateway <https://gateway.ireceptor.org/home>`_ into a Repertoire-, Sequence- or ReceptorDataset.
The differences between this importer and the :ref:`AIRR` importer are:

* This importer takes in a list of .zip files, which must contain one or more AIRR tsv files, and for each AIRR file, a corresponding metadata json file must be present.
* This importer does not require a metadata csv file for RepertoireDataset import, it is generated automatically from the metadata json files.

RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
immune receptors respectively, like antigen specificity.

AIRR rearrangement schema can be found here: https://docs.airr-community.org/en/stable/datarep/rearrangements.html

When importing a ReceptorDataset, the AIRR field cell_id is used to determine the chain pairs.


**Specification arguments:**

- path (str): This is the path to a directory **with .zip files** retrieved from the iReceptor Gateway. These .zip files should include AIRR files (with .tsv extension) and corresponding metadata.json files with matching names (e.g., for my_dataset.tsv the corresponding metadata file is called my_dataset-metadata.json). The zip files must use the .zip extension.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For RepertoireDataset labels, label_columns is ignored, metadata is discovered automatically from the metadata json.

- paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on the identifier given in the AIRR column named 'cell_id'.

- receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values are `TRA_TRB`, `TRG_TRD`, `IGH_IGL`, `IGH_IGK`. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

- import_productive (bool): Whether productive sequences (with value 'T' in column productive) should be included in the imported sequences. By default, import_productive is True.

- import_with_stop_codon (bool): Whether sequences with stop codons (with value 'T' in column stop_codon) should be included in the imported sequences. This only applies if column stop_codon is present. By default, import_with_stop_codon is False.

- import_out_of_frame (bool): Whether out of frame sequences (with value 'F' in column vj_in_frame) should be included in the imported sequences. This only applies if column vj_in_frame is present. By default, import_out_of_frame is False.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as AIRR uses the IMGT junction. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- separator (str): Column separator, for AIRR this is by default "\t".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_airr_dataset:
                format: IReceptor
                params:
                    path: path/to/zipfiles/
                    is_repertoire: True # whether to import a RepertoireDataset
                    metadata_column_mapping: # metadata column mapping AIRR: immuneML for Sequence- or ReceptorDatasetDataset
                        airr_column_name1: metadata_label1
                        airr_column_name2: metadata_label2
                    import_productive: True # whether to include productive sequences in the dataset
                    import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                    import_out_of_frame: False # whether to include out of frame sequences in the dataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even if the `sequences` column is empty (provided that other fields are as specified here)
                    import_empty_aa_sequences: False # remove all sequences with empty `sequence_aas` column
                    # Optional fields with AIRR-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    region_type: IMGT_CDR3 # what part of the sequence to import



ImmunoSEQRearrangement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data from Adaptive Biotechnologies immunoSEQ Analyzer rearrangement-level .tsv files into a
Repertoire-, or SequenceDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
antigen specificity.

The format of the files imported by this importer is described here:
https://www.adaptivebiotech.com/wp-content/uploads/2019/07/MRK-00342_immunoSEQ_TechNote_DataExport_WEB_REV.pdf
Alternatively, to import sample-level .tsv files, see :ref:`ImmunoSEQSample` import

The only difference between these two importers is which columns they load from the .tsv files.


**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the files included under the column 'filename' are imported into the RepertoireDataset. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- import_productive (bool): Whether productive sequences (with value 'In' in column frame_type) should be included in the imported sequences. By default, import_productive is True.

- import_with_stop_codon (bool): Whether sequences with stop codons (with value 'Stop' in column frame_type) should be included in the imported sequences. By default, import_with_stop_codon is False.

- import_out_of_frame (bool): Whether out of frame sequences (with value 'Out' in column frame_type) should be included in the imported sequences. By default, import_out_of_frame is False.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to check when importing. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as immunoSEQ files use the IMGT junction. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- column_mapping (dict): A mapping from immunoSEQ column names to immuneML's internal data representation. For immunoSEQ rearrangement-level files, this is by default set the values shown below in YAML format.         A custom column mapping can be specified here if necessary (for example: adding additional data fields if they are present in the file, or using alternative column names). Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. For ImmunoSEQ rearrangement import, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

          rearrangement: sequence
          amino_acid: junction_aa
          v_resolved: v_call
          j_resolved: j_call
          templates: duplicate_count

- columns_to_load (list): Specifies which subset of columns must be loaded from the file. By default, this is: [rearrangement, v_family, v_gene, v_allele, j_family, j_gene, j_allele, amino_acid, templates, frame_type, locus]

- separator (str): Column separator, for ImmunoSEQ files this is by default "\t".

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter will typically be False (import only non-empty amino acid sequences)


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_immunoseq_dataset:
                format: ImmunoSEQRearrangement
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    metadata_column_mapping: # metadata column mapping ImmunoSEQ: immuneML for SequenceDataset
                        immunoseq_column_name1: metadata_label1
                        immunoseq_column_name2: metadata_label2
                    import_productive: True # whether to include productive sequences in the dataset
                    import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                    import_out_of_frame: False # whether to include out of frame sequences in the dataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                    # Optional fields with ImmunoSEQ rearrangement-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    columns_to_load: # subset of columns to load
                    - rearrangement
                    - v_family
                    - v_gene
                    - v_resolved
                    - j_family
                    - j_gene
                    - j_resolved
                    - amino_acid
                    - templates
                    - frame_type
                    - locus
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    column_mapping: # column mapping immunoSEQ: immuneML
                        rearrangement: cdr3
                        amino_acid: cdr3_aa
                        v_resolved: v_call
                        j_resolved: j_call
                        templates: duplicate_count



ImmunoSEQSample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data from Adaptive Biotechnologies immunoSEQ Analyzer sample-level .tsv files into a
Repertoire-, or SequenceDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
antigen specificity.

The format of the files imported by this importer is described here in section 3.4.13
https://clients.adaptivebiotech.com/assets/downloads/immunoSEQ_AnalyzerManual.pdf
Alternatively, to import rearrangement-level .tsv files, see :ref:`ImmunoSEQRearrangement` import.
The only difference between these two importers is which columns they load from the .tsv files.


**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the files included under the column 'filename' are imported into the RepertoireDataset. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- import_productive (bool): Whether productive sequences (with value 'In' in column frame_type) should be included in the imported sequences. By default, import_productive is True.

- import_with_stop_codon (bool): Whether sequences with stop codons (with value 'Stop' in column frame_type) should be included in the imported sequences. By default, import_with_stop_codon is False.

- import_out_of_frame (bool): Whether out of frame sequences (with value 'Out' in column frame_type) should be included in the imported sequences. By default, import_out_of_frame is False.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as immunoSEQ files use the IMGT junction. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- column_mapping (dict): A mapping from immunoSEQ column names to immuneML's internal data representation. For immunoSEQ sample-level files, this is by default set to the values shown bellow in YAML format.         A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the file, or using alternative column names). Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. For ImmunoSEQ sample import, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

          nucleotide: sequence
          aminoAcid: junction_aa
          vGeneName: v_call
          jGeneName: j_call
          sequenceStatus: frame_type
          count (templates/reads): duplicate_count

- column_mapping_synonyms (dict): This is a column mapping that can be used if a column could have alternative names. The formatting is the same as column_mapping. If some columns specified in column_mapping are not found in the file, the columns specified in column_mapping_synonyms are instead attempted to be loaded. For immunoSEQ sample .tsv files, there is no default column_mapping_synonyms.

- columns_to_load (list): Specifies which subset of columns must be loaded from the file. By default, this is: [nucleotide, aminoAcid, count (templates/reads), vFamilyName, vGeneName, vGeneAllele, jFamilyName, jGeneName, jGeneAllele, sequenceStatus]; these are the columns from the original file that will be imported

- metadata_column_mapping (dict): Specifies metadata for Sequence- and ReceptorDatasets. This should specify a mapping similar to column_mapping where keys are immunoSEQ column names and values are the names that are internally used in immuneML as metadata fields. These metadata fields can be used as prediction labels for Sequence- and ReceptorDatasets. This parameter can also be used to specify sequence-level metadata columns for RepertoireDatasets, which can be used by reports. To set prediction label metadata for RepertoireDatasets, see metadata_file instead. For immunoSEQ sample .tsv files, there is no default metadata_column_mapping.

- separator (str): Column separator, for ImmunoSEQ files this is by default "\t".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_immunoseq_dataset:
                format: ImmunoSEQSample
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    metadata_column_mapping: # metadata column mapping ImmunoSEQ: immuneML for SequenceDataset
                        immunoseq_column_name1: metadata_label1
                        immunoseq_column_name2: metadata_label2
                    import_productive: True # whether to include productive sequences in the dataset
                    import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                    import_out_of_frame: False # whether to include out of frame sequences in the dataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                    # Optional fields with ImmunoSEQ sample-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    columns_to_load: # subset of columns to load
                    - nucleotide
                    - aminoAcid
                    - count (templates/reads)
                    - vFamilyName
                    - vGeneName
                    - vGeneAllele
                    - jFamilyName
                    - jGeneName
                    - jGeneAllele
                    - sequenceStatus
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    column_mapping: # column mapping immunoSEQ: immuneML
                        nucleotide: sequence
                        aminoAcid: junction_aa
                        vGeneName: v_call
                        jGeneName: j_call
                        sequenceStatus: frame_type
                        vFamilyName: v_family
                        jFamilyName: j_family
                        vGeneAllele: v_allele
                        jGeneAllele: j_allele
                        count (templates/reads): duplicate_count



MiXCR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data in MiXCR format into a Repertoire-, or SequenceDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets should be used when predicting values for unpaired (single-chain) immune receptors, like
antigen specificity.


**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with MiXCR files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the MiXCR files included under the column 'filename' are imported into the RepertoireDataset. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence, such as '_', are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as MiXCR format contains the trailing and leading conserved amino acids in the CDR3. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- column_mapping (dict): A mapping from MiXCR column names to immuneML's internal data representation. The columns that specify the sequences to import are handled by the region_type parameter. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the MiXCR file, or using alternative column names). Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. For MiXCR, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

        cloneCount: duplicate_count
        allVHitsWithScore: v_call
        allJHitsWithScore: j_call
        aaSeqCDR3: junction_aa
        nSeqCDR3: junction
        aaSeqCDR1: cdr1_aa
        nSeqCDR1: cdr1
        aaSeqCDR2: cdr2_aa
        nSeqCDR2: cdr2

- columns_to_load (list): Specifies which subset of columns must be loaded from the MiXCR file. By default, this is: [cloneCount, allVHitsWithScore, allJHitsWithScore, aaSeqCDR3, nSeqCDR3]

- separator (str): Column separator, for MiXCR this is by default "\t".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_mixcr_dataset:
                format: MiXCR
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                    # Optional fields with MiXCR-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    columns_to_load: # subset of columns to load, sequence columns are handled by region_type parameter
                    - cloneCount
                    - allVHitsWithScore
                    - allJHitsWithScore
                    - aaSeqCDR3
                    - nSeqCDR3
                    column_mapping: # column mapping MiXCR: immuneML
                        cloneCount: duplicate_count
                        allVHitsWithScore: v_call
                        allJHitsWithScore: j_call
                        mixcrColumnName1: metadata_label1
                        mixcrColumnName2: metadata_label2



OLGA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data generated by `OLGA <https://github.com/statbiophys/OLGA>`_ simulations into a Repertoire-, or SequenceDataset. Assumes that the columns in each
file correspond to: nucleotide sequences, amino acid sequences, v genes, j genes

Reference: Sethna, Zachary et al.
‘High-throughput immune repertoire analysis with IGoR’. Bioinformatics, (2019)
`doi.org/10.1093/bioinformatics/btz035 <https://doi.org/10.1093/bioinformatics/btz035>`_.

**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with OLGA files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. Only the OLGA files included under the column 'filename' are imported into the RepertoireDataset. SequenceDataset metadata is currently not supported.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as OLGA uses the IMGT junction. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- separator (str): Column separator, for OLGA this is by default "\t".

- column_mapping (dict): defines which columns to import from olga format: keys are the number of the columns and values are the names of the AIRR fields to be mapped to. For OLGA, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

        0: junction
        1: junction_aa
        2: v_call
        3: j_call


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_olga_dataset:
                format: OLGA
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have amino acid sequence set
                    # Optional fields with OLGA-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    columns_to_load: [0, 1, 2, 3]
                    column_mapping:
                        0: junction
                        1: junction_aa
                        2: v_call
                        3: j_call



RandomReceptorDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Returns a ReceptorDataset consisting of randomly generated sequences, which can be used for benchmarking purposes.
The sequences consist of uniformly chosen amino acids or nucleotides.


**Specification arguments:**

- receptor_count (int): The number of receptors the ReceptorDataset should contain.

- chain_1_length_probabilities (dict): A mapping where the keys correspond to different sequence lengths for chain
  1, and the values are the probabilities for choosing each sequence length. For example, to create a random
  ReceptorDataset where 40% of the sequences for chain 1 would be of length 10, and 60% of the sequences would
  have length 12, this mapping would need to be specified:

.. indent with spaces
.. code-block:: yaml

    10: 0.4
    12: 0.6

- chain_2_length_probabilities (dict): Same as chain_1_length_probabilities, but for chain 2.

- labels (dict): A mapping that specifies randomly chosen labels to be assigned to the receptors. One or multiple
  labels can be specified here. The keys of this mapping are the labels, and the values consist of another mapping
  between label classes and their probabilities. For example, to create a random ReceptorDataset with the label
  cmv_epitope where 70% of the receptors has class binding and the remaining 30% has class not_binding, the
  following mapping should be specified:

.. indent with spaces
.. code-block:: yaml

    cmv_epitope:
        binding: 0.7
        not_binding: 0.3


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_random_dataset:
                format: RandomReceptorDataset
                params:
                    receptor_count: 100 # number of random receptors to generate
                    chain_1_length_probabilities:
                        14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
                        15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
                    chain_2_length_probabilities:
                        14: 0.8 # 80% of all generated sequences for all receptors (for chain 2) will have length 14
                        15: 0.2 # 20% of all generated sequences across all receptors (for chain 2) will have length 15
                    labels:
                        epitope1: # label name
                            True: 0.5 # 50% of the receptors will have class True
                            False: 0.5 # 50% of the receptors will have class False
                        epitope2: # next label with classes that will be assigned to receptors independently of the previous label or other parameters
                            1: 0.3 # 30% of the generated receptors will have class 1
                            0: 0.7 # 70% of the generated receptors will have class 0



RandomRepertoireDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Returns a RepertoireDataset consisting of randomly generated sequences, which can be used for benchmarking purposes.
The sequences consist of uniformly chosen amino acids or nucleotides.

**Specification arguments:**

- repertoire_count (int): The number of repertoires the RepertoireDataset should contain.

- sequence_count_probabilities (dict): A mapping where the keys are the number of sequences per repertoire, and the values are the probabilities that any of the repertoires would have that number of sequences. For example, to create a random RepertoireDataset where 40% of the repertoires would have 1000 sequences, and the other 60% would have 1100 sequences, this mapping would need to be specified:

    .. indent with spaces
    .. code-block:: yaml

            1000: 0.4
            1100: 0.6

- sequence_length_probabilities (dict): A mapping where the keys correspond to different sequence lengths, and the values are the probabilities for choosing each sequence length. For example, to create a random RepertoireDataset where 40% of the sequences would be of length 10, and 60% of the sequences would have length 12, this mapping would need to be specified:

    .. indent with spaces
    .. code-block:: yaml

            10: 0.4
            12: 0.6

- labels (dict): A mapping that specifies randomly chosen labels to be assigned to the Repertoires. One or multiple labels can be specified here. The keys of this mapping are the labels, and the values consist of another mapping between label classes and their probabilities. For example, to create a random RepertoireDataset with the label CMV where 70% of the Repertoires has class cmv_positive and the remaining 30% has class cmv_negative, the following mapping should be specified:

    .. indent with spaces
    .. code-block:: yaml

            CMV:
                cmv_positive: 0.7
                cmv_negative: 0.3


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_random_dataset:
                format: RandomRepertoireDataset
                params:
                    repertoire_count: 100 # number of random repertoires to generate
                    sequence_count_probabilities:
                        10: 0.5 # probability that any of the repertoires would have 10 receptor sequences
                        20: 0.5
                    sequence_length_probabilities:
                        10: 0.5 # probability that any of the receptor sequences would be 10 amino acids in length
                        12: 0.5
                    labels: # randomly assigned labels (only useful for simple benchmarking)
                        cmv:
                            True: 0.5 # probability of value True for label cmv to be assigned to any repertoire
                            False: 0.5



RandomSequenceDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Returns a SequenceDataset consisting of randomly generated sequences, which can be used for benchmarking purposes.
The sequences consist of uniformly chosen amino acids or nucleotides.


**Specification arguments:**

- sequence_count (int): The number of sequences the SequenceDataset should contain.

- length_probabilities (dict): A mapping where the keys correspond to different sequence lengths and the values
  are the probabilities for choosing each sequence length. For example, to create a random SequenceDataset where
  40% of the sequences would be of length 10, and 60% of the sequences would have length 12, this mapping would
  need to be specified:

.. indent with spaces
.. code-block:: yaml

        10: 0.4
        12: 0.6

- labels (dict): A mapping that specifies randomly chosen labels to be assigned to the sequences. One or multiple
  labels can be specified here. The keys of this mapping are the labels, and the values consist of another mapping
  between label classes and their probabilities. For example, to create a random SequenceDataset with the label
  cmv_epitope where 70% of the sequences has class binding and the remaining 30% has class not_binding, the
  following mapping should be specified:

.. indent with spaces
.. code-block:: yaml

        cmv_epitope:
            binding: 0.7
            not_binding: 0.3

- region_type (str): which region_type to assign to all randomly generated sequences


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_random_dataset:
                format: RandomSequenceDataset
                params:
                    sequence_count: 100 # number of random sequences to generate
                    length_probabilities:
                        14: 0.8 # 80% of all generated sequences for all sequences will have length 14
                        15: 0.2 # 20% of all generated sequences across all sequences will have length 15
                    labels:
                        epitope1: # label name
                            True: 0.5 # 50% of the sequences will have class True
                            False: 0.5 # 50% of the sequences will have class False
                        epitope2: # next label with classes that will be assigned to sequences independently of the previous label or other parameters
                            1: 0.3 # 30% of the generated sequences will have class 1
                            0: 0.7 # 70% of the generated sequences will have class 0



TenxGenomics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data from the 10x Genomics Cell Ranger analysis pipeline into a Repertoire-, Sequence- or ReceptorDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
immune receptors respectively, like antigen specificity.

.. note::

    The 10xGenomics Cell Ranger VDJ pipeline also directly exports data in AIRR format ('airr_rearrangement.tsv',
    see https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-5p-outputs-overview-vdj).
    If possible, we highly recommend directly using the AIRR formatted files as input for immuneML.


If AIRR files are not available, this importer may be used to import data from Contig annotation CSV files
('all_contig_annotations.csv' or 'filtered_contig_annotations.csv') as described here: https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-5p-outputs-annotations-vdj#contig-annotation-csv

It is recommended to run :py:obj:`~immuneML.preprocessing.filters.DuplicateSequenceFilter.DuplicateSequenceFilter` to
collapse together clonotypes when importing as a RepertoireDataset, and in some cases for Sequence- and ReceptorDatasets.

Note that for pairing together Receptor chains, the column named 'barcode' is used.


**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with 10xGenomics files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on the identifier given in the 10xGenomics column named 'clonotype_id'.

- receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor.  Valid values are `TRA_TRB`, `TRG_TRD`, `IGH_IGL`, `IGH_IGK`. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

- import_productive (bool): Whether productive sequences (with value 'True' in column productive) should be included in the imported sequences. By default, import_productive is True.

- import_unproductive (bool): Whether productive sequences (with value 'Fale' in column productive) should be included in the imported sequences. By default, import_unproductive is False.

- import_unknown_productivity (bool): Whether sequences with unknown productivity (missing or 'NA' value in column productive) should be included in the imported sequences. By default, import_unknown_productivity is True.

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as 10xGenomics uses IMGT junction as CDR3. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- column_mapping (dict): A mapping from 10xGenomics column names to immuneML's internal data representation. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the 10xGenomics file, or using alternative column names). Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. For 10xGenomics, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

            cdr3: junction
            cdr3_nt: junction_aa
            v_gene: v_call
            j_gene: j_call
            umis: duplicate_count
            clonotype_id: cell_id
            consensus_id: sequence_id

- column_mapping_synonyms (dict): This is a column mapping that can be used if a column could have alternative names. The formatting is the same as column_mapping. If some columns specified in column_mapping are not found in the file, the columns specified in column_mapping_synonyms are instead attempted to be loaded. For 10xGenomics format, there is no default column_mapping_synonyms.

- separator (str): Column separator, for 10xGenomics this is by default ",".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_10x_dataset:
                format: 10xGenomics
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                    receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have amino acid sequence set
                    # Optional fields with 10xGenomics-specific defaults, only change when different behavior is required:
                    separator: "," # column separator
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    column_mapping: # column mapping 10xGenomics: immuneML
                        cdr3: junction_aa
                        cdr3_nt: junction
                        v_gene: v_call
                        j_gene: j_call
                        umis: duplicate_count
                        clonotype_id: cell_id
                        consensus_id: sequence_id



VDJdb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Imports data in VDJdb format into a Repertoire-, Sequence- or ReceptorDataset.
RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
immune receptors respectively, like antigen specificity.


**Specification arguments:**

- path (str): For RepertoireDatasets, this is the path to a directory with VDJdb files to import. For Sequence- or ReceptorDatasets this path may either be the path to the file to import, or the path to the folder locating one or multiple files with .tsv, .csv or .txt extensions. By default path is set to the current working directory.

- is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

- metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file. This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions. For setting Sequence- or ReceptorDataset labels, metadata_file is ignored, use label_columns instead.

- label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. By default, label_columns for VDJdbImport are [Epitope, Epitope gene, Epitope species]. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For setting RepertoireDataset labels, label_columns is ignored, use metadata_file instead.

- paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on the identifier given in the VDJdb column named 'complex.id'.

- receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values are `TRA_TRB`, `TRG_TRD`, `IGH_IGL`, `IGH_IGK`. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

- import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

- import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

- import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

- region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as VDJdb uses IMGT junction as CDR3. Specifying any other value will result in importing the sequences as they are. Valid values are `IMGT_CDR1`, `IMGT_CDR2`, `IMGT_CDR3`, `IMGT_FR1`, `IMGT_FR2`, `IMGT_FR3`, `IMGT_FR4`, `IMGT_JUNCTION`, `FULL_SEQUENCE`.

- column_mapping (dict): A mapping from VDJdb column names to immuneML's internal data representation. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the VDJdb file, or using alternative column names). Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.. For VDJdb, this is by default set to:

    .. indent with spaces
    .. code-block:: yaml

            V: v_call
            J: j_call
            CDR3: junction_aa
            complex.id: cell_id
            Gene: locus

- separator (str): Column separator, for VDJdb this is by default "\t".


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        datasets:
            my_vdjdb_dataset:
                format: VDJdb
                params:
                    path: path/to/files/
                    is_repertoire: True # whether to import a RepertoireDataset
                    metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                    paired: False # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                    receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                    import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                    import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                    import_empty_aa_sequences: False # filter out sequences if they don't have amino acid sequence set
                    # Optional fields with VDJdb-specific defaults, only change when different behavior is required:
                    separator: "\t" # column separator
                    region_type: IMGT_CDR3 # what part of the sequence to import
                    column_mapping: # column mapping VDJdb: immuneML
                        V: v_call
                        J: j_call
                        CDR3: junction_aa
                        complex.id: sequence_id
                        Gene: chain
                        Epitope: epitope
                        Epitope gene: epitope_gene
                        Epitope species: epitope_species


