import pickle
import random as rn
from glob import glob

import pandas as pd

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.IRISImportParams import IRISImportParams
from source.IO.dataset_import.IRISSequenceImport import IRISSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.util.ImportHelper import ImportHelper


class IRISImport(DataImport):
    """
    Imports data from IRIS format into a ReceptorDataset or SequenceDataset depending on the value of "paired" parameter
    (if metadata file is not defined) or to RepertoireDataset (a set of repertoires consisting of a list of receptor sequences) if the
    metadata file is defined.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_iris_dataset:
            format: IRIS
            params:
                sequence_file_size: 1000 # number of sequences / receptors per file as stored internally by ImmuneML in ImmuneML format - not visible to users
                paired: True
                import_dual_chains: True
                import_all_gene_combinations: False
                batch_size: 4
                separator: "\\t"
                extra_columns_to_load: []

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

        filenames = glob(iris_params.path + "*.tsv")
        file_index = 0
        dataset_filenames = []

        for index, filename in enumerate(filenames):
            items = IRISSequenceImport.import_items(filename, paired=iris_params.paired,
                                                    all_dual_chains=iris_params.import_dual_chains,
                                                    all_genes=iris_params.import_all_gene_combinations)

            while len(items) > iris_params.sequence_file_size or (index == len(filenames) - 1 and len(items) > 0):
                dataset_filenames.append(iris_params.result_path + "batch_{}.pickle".format(file_index))
                ImportHelper.store_sequence_items(dataset_filenames, items, iris_params.sequence_file_size)
                items = items[iris_params.sequence_file_size:]
                file_index += 1

        return ReceptorDataset(filenames=dataset_filenames, file_size=iris_params.sequence_file_size, name=dataset_name) if iris_params.paired \
            else SequenceDataset(filenames=dataset_filenames, file_size=iris_params.sequence_file_size, name=dataset_name)


