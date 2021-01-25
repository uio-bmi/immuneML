import random as rn

import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.IRISImportParams import IRISImportParams
from immuneML.IO.dataset_import.IRISSequenceImport import IRISSequenceImport
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.util.ImportHelper import ImportHelper


class IRISImport(DataImport):
    """
    Imports data in Immune Receptor Information System (IRIS) format into a Repertoire-, Sequence- or ReceptorDataset.

    Arguments:

        path (str): This is the path to a directory with IRIS files to import. By default path is set to the current working directory.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or
        ReceptorDataset. By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
        This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
        Sequence- or ReceptorDataset metadata is currently not supported.

        paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a
        SequenceDataset (paired = False) or a ReceptorDataset (paired = True). By default, paired = True.

        receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor.
        Valid values for receptor_chains are the names of the ChainPair enum.

        import_dual_chains (bool): Whether to import the dual chains, as denoted by the suffix (2) in the Chain, V gene
        and J gene columns. If this is True and a ReceptorDataset is imported, all possible combinations of chains (up to 4)
        are imported as individual receptors. By default import_dual_chains is True.

        import_all_gene_combinations (bool):  Whether to import all possible genes when multiple genes are present in the
        V gene and J gene columns. If this is False, a random gene is chosen. If this is true and a ReceptorDataset
        is imported, all possible combinations of chains are imported as individual receptors. By default import_all_gene_combinations
        is False.

        import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False.
        By default, import_empty_nt_sequences is set to True.

        import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on
        amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

        extra_columns_to_load (list): Additional columns that should be loaded, apart from the default columns
        (Clonotype ID and any Chain, V gene and J gene columns).

        separator (str): Column separator, for IRIS this is by default "\\t".


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_iris_dataset:
            format: IRIS
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                paired: True # whether to import SequenceDataset (False) or ReceptorDataset (True) when is_repertoire = False
                receptor_chains: TRA_TRB # what chain pair to import for a ReceptorDataset
                import_dual_chains: True
                import_all_gene_combinations: False
                import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
                import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
                extra_columns_to_load:
                - extra_col1
                - extra_col2
                # Optional fields with IRIS-specific defaults, only change when different behavior is required:
                separator: "\\t" # column separator
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        iris_params = IRISImportParams.build_object(**params)

        dataset = ImportHelper.load_dataset_if_exists(params, iris_params, dataset_name)
        if dataset is None:
            if iris_params.is_repertoire:
                dataset = ImportHelper.import_repertoire_dataset(IRISImport, iris_params, dataset_name)
            else:
                dataset = IRISImport.load_sequence_dataset(params, dataset_name)

        return dataset

    @staticmethod
    def _load_gene(identifier):
        return identifier.split("*", 1)[0]

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params):

        subframes = []

        chain_dups_to_process = ("1", "2") if params.import_dual_chains is True else ("1")

        for chain in params.receptor_chains.value:
            for chain_dup in chain_dups_to_process:
                subframe_dict = {"cell_ids": df["Clonotype ID"],
                                 "sequence_aas": df[f"Chain: {chain} ({chain_dup})"],
                                 "v_genes": df[f"{chain} - V gene ({chain_dup})"],
                                 "j_genes": df[f"{chain} - J gene ({chain_dup})"],
                                 "chains": Chain.get_chain(chain).value}
                if params.extra_columns_to_load is not None:
                    for extra_col in params.extra_columns_to_load:
                        subframe_dict[extra_col] = df[extra_col]
                subframes.append(pd.DataFrame(subframe_dict))

        df = pd.concat(subframes, axis=0)
        df.dropna(subset=["sequence_aas", "v_genes", "j_genes"], inplace=True)

        df.reset_index(drop=True, inplace=True)

        if params.import_all_gene_combinations:
            df = IRISImport.import_all_gene_combinations(df)
        else:
            for gene_column in ("v_genes", "j_genes"):
                processed_names = [IRISImport._load_gene(rn.choice(raw_v_string.split(" | "))) for raw_v_string in df[gene_column]]
                df[gene_column] = processed_names

        ImportHelper.drop_empty_sequences(df, params.import_empty_aa_sequences, params.import_empty_nt_sequences)

        return df

    @staticmethod
    def import_all_gene_combinations(df: pd.DataFrame):
        dict_repr = df.to_dict()

        for gene_column in ("v_genes", "j_genes"):
            other_columns = list(df.columns)
            other_columns.remove(gene_column)

            for idx in list(dict_repr[gene_column].keys()):
                all_genes = dict_repr[gene_column][idx]
                unique_genes = set([IRISImport._load_gene(full_gene_str) for full_gene_str in all_genes.split(" | ")])

                if len(unique_genes) == 1:
                    dict_repr[gene_column][idx] = unique_genes.pop()
                else:
                    sub_idx = 0
                    while unique_genes:
                        dict_repr[gene_column][f"{idx}|{sub_idx}"] = unique_genes.pop()
                        for col in other_columns:
                            dict_repr[col][f"{idx}|{sub_idx}"] = dict_repr[col][idx]

                        sub_idx += 1

        df = pd.DataFrame(dict_repr)
        df = df[~(df.v_genes.str.contains(" | ") | df.j_genes.str.contains(" | "))]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def alternative_load_func(filepath, params):
        first_chain, second_chain = params.receptor_chains.value

        usecols = ["Clonotype ID",
                   f"Chain: {first_chain} (1)", f"{first_chain} - V gene (1)",
                   f"{first_chain} - J gene (1)",
                   f"Chain: {first_chain} (2)", f"{first_chain} - V gene (2)",
                   f"{first_chain} - J gene (2)",
                   f"Chain: {second_chain} (1)", f"{second_chain} - V gene (1)",
                   f"{second_chain} - J gene (1)",
                   f"Chain: {second_chain} (2)", f"{second_chain} - V gene (2)",
                   f"{second_chain} - J gene (2)"]

        if type(params.extra_columns_to_load) is list:
            usecols = usecols + params.extra_columns_to_load

        df = pd.read_csv(filepath, sep=params.separator, iterator=False, dtype=str, usecols=usecols)
        return df

    @staticmethod
    def load_sequence_dataset(params: dict, dataset_name: str) -> Dataset:

        iris_params = IRISImportParams.build_object(**params)

        filenames = ImportHelper.get_sequence_filenames(iris_params.path, dataset_name)
        file_index = 0
        dataset_filenames = []

        for index, filename in enumerate(filenames):
            items = IRISSequenceImport.import_items(filename, paired=iris_params.paired,
                                                    all_dual_chains=iris_params.import_dual_chains,
                                                    all_genes=iris_params.import_all_gene_combinations)

            while len(items) > iris_params.sequence_file_size or (index == len(filenames) - 1 and len(items) > 0):
                dataset_filenames.append(iris_params.result_path / "batch_{}.pickle".format(file_index))
                ImportHelper.store_sequence_items(dataset_filenames, items, iris_params.sequence_file_size)
                items = items[iris_params.sequence_file_size:]
                file_index += 1

        return ReceptorDataset(filenames=dataset_filenames, file_size=iris_params.sequence_file_size, name=dataset_name) if iris_params.paired \
            else SequenceDataset(filenames=dataset_filenames, file_size=iris_params.sequence_file_size, name=dataset_name)
