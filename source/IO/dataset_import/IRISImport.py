import pickle
from glob import glob
import pandas as pd
import random as rn

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.IO.sequence_import.IRISSequenceImport import IRISSequenceImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.ImportHelper import ImportHelper


class IRISImport(DataImport):
    """
    Imports data from IRIS format into a ReceptorDataset or SequenceDataset depending on the value of "paired" parameter
    (if metadata file is not defined) or to RepertoireDataset (a set of repertoires consisting of a list of receptor sequences) if the
    metadata file is defined.
    """

    @staticmethod
    def import_dataset(params: DatasetImportParams) -> Dataset:
        if params.metadata_file is not None:
            dataset = IRISImport.load_repertoire_dataset(params)
        else:
            dataset = IRISImport.load_sequence_dataset(params)
        return dataset

    @staticmethod
    def load_repertoire_dataset(params: DatasetImportParams) -> Dataset:
        return ImportHelper.import_repertoire_dataset(IRISImport.preprocess_repertoire, params)

    @staticmethod
    def _load_gene(identifier):
        return identifier.split("*", 1)[0].replace("TRA", "").replace("TRB", "")

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:
        # todo make compatible with nucleotide sequences (no use case yet)
        # todo make compatible with gamma/delta and BCR (no use case yet)
        df = ImportHelper.load_repertoire_as_dataframe(metadata, params, alternative_load_func=IRISImport.load_iris_dataframe)

        subframes = []

        chain_dups_to_process = ("1", "2") if params.misc["all_dual_chains"] is True else ("1")

        for chain in ("A", "B"):
            for chain_dup in chain_dups_to_process:
                subframes.append(pd.DataFrame({"cell_ids": df["Clonotype ID"],
                                               "sequence_aas": df[f"Chain: TR{chain} ({chain_dup})"],
                                               "v_genes": df[f"TR{chain} - V gene ({chain_dup})"],
                                               "j_genes": df[f"TR{chain} - J gene ({chain_dup})"],
                                               "chains": Chain(chain).value}))

        df = pd.concat(subframes, axis=0)
        df.dropna(subset=["sequence_aas", "v_genes", "j_genes"], inplace=True)

        df.reset_index(drop=True, inplace=True)

        if params.misc["all_genes"]:
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

        else:
            for gene_column in ("v_genes", "j_genes"):
                processed_names = [IRISImport._load_gene(rn.choice(raw_v_string.split(" | "))) for raw_v_string in df[gene_column]]
                df[gene_column] = processed_names

        return df

    @staticmethod
    def load_iris_dataframe(filepath, params):
        df = pd.read_csv(filepath, sep=params.separator, iterator=False, dtype=str, usecols=["Clonotype ID",
                                                                                 "Chain: TRA (1)", "TRA - V gene (1)", "TRA - J gene (1)",
                                                                                 "Chain: TRA (2)", "TRA - V gene (2)", "TRA - J gene (2)",
                                                                                 "Chain: TRB (1)", "TRB - V gene (1)", "TRB - J gene (1)",
                                                                                 "Chain: TRB (2)", "TRB - V gene (2)", "TRB - J gene (2)"])
        return df

    @staticmethod
    def load_sequence_dataset(params: DatasetImportParams) -> Dataset:

        filenames = glob(params.path + "*.tsv")
        file_index = 0
        dataset_filenames = []

        for index, filename in enumerate(filenames):
            items = IRISSequenceImport.import_items(filename, paired=params.paired,
                                                    all_dual_chains=params.misc["all_dual_chains"],
                                                    all_genes=params.misc["all_genes"])

            while len(items) > params.file_size or (index == len(filenames)-1 and len(items) > 0):
                dataset_filenames.append(params.result_path + "batch_{}.pickle".format(file_index))
                IRISImport.store_items(dataset_filenames, items, params.file_size)
                items = items[params.file_size:]
                file_index += 1

        return ReceptorDataset(filenames=dataset_filenames, file_size=params.file_size) if params.paired \
            else SequenceDataset(filenames=dataset_filenames, file_size=params.file_size)

    @staticmethod
    def store_items(dataset_filenames: list, items: list, file_size: int):
        with open(dataset_filenames[-1], "wb") as file:
            pickle.dump(items[:file_size], file)
