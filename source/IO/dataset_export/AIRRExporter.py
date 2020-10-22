# quality: gold
import logging
import math
from enum import Enum

import airr
import pandas as pd
from typing import List

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.Receptor import Receptor
from source.data_model.receptor.RegionType import RegionType
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.util.PathBuilder import PathBuilder


class AIRRExporter(DataExporter):
    """
    Exports a RepertoireDataset of Repertoires in AIRR format.
    This exporter does not support Sequence- or ReceptorDatasets

    Things to note:
        - one filename_prefix is given, which is combined with the Repertoire identifiers
        for the filenames, to create one file per Repertoire
        - 'counts' is written into the field 'duplicate_counts'
        - 'sequence_identifiers' is written both into the fields 'sequence_id' and 'rearrangement_id'

    """

    @staticmethod
    def export(dataset: Dataset, path, region_type=RegionType.IMGT_CDR3):
        # if not isinstance(dataset, RepertoireDataset):
        #     raise ValueError(f"AIRRExporter: dataset {dataset.name} is a {type(dataset).__name__}, but only repertoire dataset export is currently "
        #                     f"supported for AIRR format.")
        # else:
        PathBuilder.build(path)

        if isinstance(dataset, RepertoireDataset):
            repertoire_path = PathBuilder.build(f"{path}repertoires/")

            for index, repertoire in enumerate(dataset.repertoires):
                df = AIRRExporter._repertoire_to_dataframe(repertoire, region_type)
                airr.dump_rearrangement(df, f"{repertoire_path}{repertoire.identifier}.tsv")

            AIRRExporter.export_updated_metadata(dataset, path)
        else:

            index = 1
            file_count = math.ceil(dataset.get_example_count() / dataset.file_size)

            for batch in dataset.get_batch():
                filename = f"{path}batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.tsv"

                if isinstance(dataset, ReceptorDataset):
                    df = AIRRExporter._receptors_to_dataframe(batch, region_type)
                else:
                    df = AIRRExporter._sequences_to_dataframe(batch, region_type)

                airr.dump_rearrangement(df, filename)

                index += 1

    @staticmethod
    def get_sequence_field(region_type):
        if region_type == RegionType.IMGT_CDR3:
            return "cdr3"
        else:
            return "sequence"

    @staticmethod
    def get_sequence_aa_field(region_type):
        if region_type == RegionType.IMGT_CDR3:
            return "cdr3_aa"
        else:
            return "sequence_aa"

    @staticmethod
    def export_updated_metadata(dataset: RepertoireDataset, result_path: str):
        df = pd.read_csv(dataset.metadata_file, comment=Constants.COMMENT_SIGN)
        identifiers = df["repertoire_identifier"].values.tolist() if "repertoire_identifier" in df.columns else dataset.get_example_ids()
        df["filename"] = [f"{item}.tsv" for item in identifiers]
        df.to_csv(f"{result_path}metadata.csv", index=False)

    @staticmethod
    def _repertoire_to_dataframe(repertoire: Repertoire, region_type):
        # get all fields (including custom fields)
        df = pd.DataFrame({key: repertoire.get_attribute(key) for key in set(repertoire.fields)})

        # rename mandatory fields for airr-compliance
        mapper = {"sequence_identifiers": "sequence_id",
                  "v_genes": "v_call",
                  "j_genes": "j_call",
                  "chains": "locus",
                  "counts": "duplicate_count"}

        mapper["sequences"] = AIRRExporter.get_sequence_field(region_type)
        mapper["sequence_aas"] = AIRRExporter.get_sequence_aa_field(region_type)

        df = df.rename(mapper=mapper, axis="columns")

        if "locus" in df.columns:
            chain_conversion_dict = {Chain.ALPHA: "TRA",
                                     Chain.BETA: "TRB",
                                     Chain.HEAVY: "IGH",
                                     Chain.LIGHT: "IGL",
                                     None: ''}
            df["locus"] = [chain_conversion_dict[chain] for chain in df["locus"]]

        return df

    @staticmethod
    def _receptors_to_dataframe(receptors: List[Receptor], region_type):
        # for receptor in receptors:

        sequences = [(receptor.get_chain(receptor.get_chains()[0]), receptor.get_chain(receptor.get_chains()[1])) for receptor in receptors]
        sequences = [item for sublist in sequences for item in sublist]

        return AIRRExporter._sequences_to_dataframe(sequences, region_type)

    @staticmethod
    def _sequences_to_dataframe(sequences: List[ReceptorSequence], region_type):
        sequence_field = AIRRExporter.get_sequence_field(region_type)
        sequence_aa_field = AIRRExporter.get_sequence_aa_field(region_type)

        main_data_dict = {"sequence_id": [], sequence_field: [], sequence_aa_field: []}
        attributes_dict = {"chain": [], "v_gene": [], "j_gene": [], "count": []}

        for i, sequence in enumerate(sequences):
            main_data_dict["sequence_id"].append(sequence.identifier)
            main_data_dict[sequence_field].append(sequence.nucleotide_sequence)
            main_data_dict[sequence_aa_field].append(sequence.amino_acid_sequence)

            # add custom params of this receptor sequence to attributes dict
            if sequence.metadata is not None and sequence.metadata.custom_params is not None:
                for custom_param in sequence.metadata.custom_params:
                    if custom_param not in attributes_dict:
                        attributes_dict[custom_param] = [None for i in range(i)]

            for attribute in attributes_dict.keys():
                try:
                    attr_value = sequence.get_attribute(attribute)
                    if isinstance(attr_value, Enum):
                        attr_value = attr_value.value
                    attributes_dict[attribute].append(attr_value)
                except KeyError:
                    attributes_dict[attribute].append(None)

        df = pd.DataFrame({**attributes_dict, **main_data_dict})
        df.rename(columns={"v_gene": "v_call", "j_gene": "j_call", "chain": "locus", "count": "duplicate_count"}, inplace=True)

        return df
