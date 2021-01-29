from typing import List

import pandas as pd

from immuneML.IO.dataset_export.PickleExporter import PickleExporter
from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.ReceptorBuilder import ReceptorBuilder
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.PathBuilder import PathBuilder
from scripts.specification_util import update_docs_per_mapping


class SingleLineReceptorImport(DataImport):
    """
    Imports data from a tabular file (where each line contains a pair of immune receptor sequences) into a ReceptorDataset.
    If you instead want to import a ReceptorDataset from a tabular file that contains one receptor sequence per line,
    see :ref:`Generic` import.


    Arguments:

        path (str): Required parameter. This is the path to a directory with files to import.

        receptor_chains (str): Required parameter. Determines which pair of chains to import for each Receptor.
        Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.

        import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False.
        By default, import_empty_nt_sequences is set to True.

        import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on
        amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

        region_type (str): Which part of the sequence to import. When IMGT_CDR3 is specified, immuneML assumes the IMGT
        junction (including leading C and trailing Y/F amino acids) is used in the input file, and the first and last
        amino acids will be removed from the sequences to retrieve the IMGT CDR3 sequence. Specifying any other value
        will result in importing the sequences as they are.
        Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

        column_mapping (dict): A mapping where the keys are the column names in the input file, and the values must be
        mapped to the following fields: <chain>_amino_acid_sequence, <chain>_nucleotide_sequence, <chain>_v_gene,
        <chain>_j_gene, identifier, epitope.
        The possible names that can be filled in for <chain> are given in :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`
        Any column namme other than the sequence, v/j genes and identifier will be set as metadata fields to the
        Receptors, and can subsequently be used as labels in immuneML instructions.
        For TCR alpha-beta receptor import, a column mapping could for example look like this:

        .. indent with spaces
        .. code-block:: yaml

                cdr3_a_aa: alpha_amino_acid_sequence
                cdr3_b_aa: beta_amino_acid_sequence
                cdr3_a_nucseq: alpha_nucleotide_sequence
                cdr3_b_nucseq: beta_nucleotide_sequence
                v_a_gene: alpha_v_gene
                v_b_gene: beta_v_gene
                j_a_gene: alpha_j_gene
                j_b_gene: beta_j_gene
                clone_id: identifier
                epitope: epitope # metadata field

        columns_to_load (list): Optional; specifies which columns to load from the input file. This may be useful if
        the input files contain many unused columns. If no value is specified, all columns are loaded.

        separator (str): Required parameter. Column separator, for example "\\t" or ",".

        organism (str): The organism that the receptors came from. This will be set as a parameter in the ReceptorDataset object.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_receptor_dataset:
            format: SingleLineReceptor
            params:
                path: path/to/files/
                receptor_chains: TRA_TRB # what chain pair to import
                separator: "\\t" # column separator
                import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                region_type: IMGT_CDR3 # what part of the sequence to import
                columns_to_load: # which subset of columns to load from the file
                - subject
                - epitope
                - count
                - v_a_gene
                - j_a_gene
                - cdr3_a_aa
                - v_b_gene
                - j_b_gene
                - cdr3_b_aa
                - clone_id
                column_mapping: # column mapping file: immuneML
                    cdr3_a_aa: alpha_amino_acid_sequence
                    cdr3_b_aa: beta_amino_acid_sequence
                    cdr3_a_nucseq: alpha_nucleotide_sequence
                    cdr3_b_nucseq: beta_nucleotide_sequence
                    v_a_gene: alpha_v_gene
                    v_b_gene: beta_v_gene
                    j_a_gene: alpha_j_gene
                    j_b_gene: beta_j_gene
                    clone_id: identifier
                    epitope: epitope
                    organism: mouse

    """

    @staticmethod
    def import_dataset(params, dataset_name: str) -> ReceptorDataset:
        generic_params = DatasetImportParams.build_object(**params)

        filenames = ImportHelper.get_sequence_filenames(generic_params.path, dataset_name)

        PathBuilder.build(generic_params.result_path, warn_if_exists=True)

        dataset = SingleLineReceptorImport._import_from_files(filenames, generic_params)
        dataset.name = dataset_name
        dataset.labels = ImportHelper.extract_sequence_dataset_params(params=generic_params)

        PickleExporter.export(dataset, generic_params.result_path)

        return dataset

    @staticmethod
    def _import_from_files(filenames: List[str], generic_params: DatasetImportParams) -> ReceptorDataset:
        elements = []

        for file in filenames:
            df = pd.read_csv(file, sep=generic_params.separator, usecols=generic_params.columns_to_load)
            df.dropna()
            df.drop_duplicates()
            df.rename(columns=generic_params.column_mapping, inplace=True)

            if "alpha_amino_acid_sequence" in df:
                df["alpha_amino_acid_sequence"] = df["alpha_amino_acid_sequence"].str[1:-1]
            if "beta_amino_acid_sequence" in df:
                df["beta_amino_acid_sequence"] = df["beta_amino_acid_sequence"].str[1:-1]
            if "alpha_nucleotide_sequence" in df:
                df["alpha_nucleotide_sequence"] = df["alpha_nucleotide_sequence"].str[3:-3]
            if "beta_nucleotide_sequence" in df:
                df["beta_nucleotide_sequence"] = df["beta_nucleotide_sequence"].str[3:-3]

            chain_vals = [ch for ch in generic_params.receptor_chains.value]
            chain_names = [Chain.get_chain(ch).name.lower() for ch in generic_params.receptor_chains.value]

            for chain_name in chain_names:
                df = SingleLineReceptorImport.make_gene_columns(df, ["v", "j"], chain_name)

            for index, row in df.iterrows():
                sequences = {chain_vals[i]: ReceptorSequence(amino_acid_sequence=row[
                                     chain_name + "_amino_acid_sequence"] if chain_name + "_amino_acid_sequence" in row else None,
                                                  nucleotide_sequence=row[
                                                      chain_name + "_nucleotide_sequence"] if chain_name + "_nucleotide_sequence" in row else None,
                                                  metadata=SequenceMetadata(
                                                      v_gene=row[f"{chain_name}_v_gene"], v_allele=row[f"{chain_name}_v_allele"],
                                                      v_subgroup=row[f'{chain_name}_v_subgroup'],
                                                      j_gene=row[f"{chain_name}_j_gene"], j_allele=row[f"{chain_name}_j_allele"],
                                                      j_subgroup=row[f'{chain_name}_j_subgroup'],
                                                      chain=chain_name, count=row["count"], region_type=generic_params.region_type.value))
                             for i, chain_name in enumerate(chain_names)}

                elements.append(ReceptorBuilder.build_object(sequences, row["identifier"],
                                                             {key: row[key] for key in row.keys()
                                                              if all(item not in key for item in
                                                                     ["v_gene", 'j_gene', "count", "identifier"] + chain_names)}))

        return ReceptorDataset.build(elements, generic_params.sequence_file_size, generic_params.result_path)

    @staticmethod
    def make_gene_columns(df: pd.DataFrame, genes: list, chain_name=None):
        for gene in genes:
            column_name = f"{gene}_gene" if chain_name is None else f"{chain_name}_{gene}_gene"
            if column_name in df.columns:
                df[column_name.replace("gene", "allele")] = df[column_name]
                df[column_name] = [item.split("*")[0] for item in df[column_name]]
                df[column_name.replace("gene", "subgroup")] = [item.split("-")[0] for item in df[column_name]]
        return df

    @staticmethod
    def get_documentation():
        doc = str(SingleLineReceptorImport.__doc__)

        valid_chain_names = str([item.name for item in Chain])[1:-1].replace("'", "`")
        valid_chain_pair_names = str([item.name for item in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "The possible names that can be filled in for <chain> are given in :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`":
                f"The possible names that can be filled in for <chain> are: {valid_chain_names}.",
            "Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.":
                f"Valid values for receptor_chains are: {valid_chain_pair_names}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",

        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc