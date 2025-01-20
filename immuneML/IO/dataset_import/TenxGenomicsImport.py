import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.SequenceParams import ChainPair, RegionType
from immuneML.data_model.SequenceSet import Repertoire
from scripts.specification_util import update_docs_per_mapping


class TenxGenomicsImport(DataImport):
    """
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

    - receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor.  Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

    - import_productive (bool): Whether productive sequences (with value 'True' in column productive) should be included in the imported sequences. By default, import_productive is True.

    - import_unproductive (bool): Whether productive sequences (with value 'Fale' in column productive) should be included in the imported sequences. By default, import_unproductive is False.

    - import_unknown_productivity (bool): Whether sequences with unknown productivity (missing or 'NA' value in column productive) should be included in the imported sequences. By default, import_unknown_productivity is True.

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as 10xGenomics uses IMGT junction as CDR3. Specifying any other value will result in importing the sequences as they are. Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

    - column_mapping (dict): A mapping from 10xGenomics column names to immuneML's internal data representation. A custom column mapping can be specified here if necessary (for example; adding additional data fields if they are present in the 10xGenomics file, or using alternative column names). Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet). For 10xGenomics, this is by default set to:

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

    """

    @staticmethod
    def get_documentation():
        doc = str(TenxGenomicsImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by the AIRR standard (AIRRSequenceSet)": f"Valid immuneML fields that can be specified here by `the AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc


