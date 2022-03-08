# quality: gold
import math
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import List

import airr
import pandas as pd

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.Receptor import Receptor
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.util.NumpyHelper import NumpyHelper
from immuneML.util.PathBuilder import PathBuilder


class AIRRExporter(DataExporter):
    """
    Exports a RepertoireDataset of Repertoires in AIRR format.

    Things to note:
        - one filename_prefix is given, which is combined with the Repertoire identifiers
        for the filenames, to create one file per Repertoire
        - 'counts' is written into the field 'duplicate_counts'
        - 'sequence_identifiers' is written both into the fields 'sequence_id' and 'rearrangement_id'

    """

    @staticmethod
    def export(dataset: Dataset, path: Path, number_of_processes: int = 1):
        PathBuilder.build(path)

        if isinstance(dataset, RepertoireDataset):
            repertoire_folder = "repertoires/"
            repertoire_path = PathBuilder.build(path / repertoire_folder)

            with Pool(processes=number_of_processes) as pool:
                pool.starmap(AIRRExporter.export_repertoire, [(repertoire, repertoire_path) for repertoire in dataset.repertoires])

            AIRRExporter.export_updated_metadata(dataset, path, repertoire_folder)
        else:

            index = 1
            file_count = math.ceil(dataset.get_example_count() / dataset.file_size)

            for batch in dataset.get_batch():
                filename = path / f"batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.tsv"

                if isinstance(dataset, ReceptorDataset):
                    df = AIRRExporter._receptors_to_dataframe(batch)
                else:
                    df = AIRRExporter._sequences_to_dataframe(batch)

                df = AIRRExporter._postprocess_dataframe(df)
                airr.dump_rearrangement(df, filename)

                index += 1

    @staticmethod
    def export_repertoire(repertoire: Repertoire, repertoire_path: Path):
        df = AIRRExporter._repertoire_to_dataframe(repertoire)
        df = AIRRExporter._postprocess_dataframe(df)
        output_file = repertoire_path / f"{repertoire.data_filename.stem}.tsv"
        airr.dump_rearrangement(df, str(output_file))

    @staticmethod
    def get_sequence_field(region_type):
        if region_type == RegionType.IMGT_CDR3:
            return "cdr3"
        elif region_type == RegionType.IMGT_JUNCTION:
            return "junction"
        else:
            return "sequence"

    @staticmethod
    def get_sequence_aa_field(region_type):
        return f"{AIRRExporter.get_sequence_field(region_type)}_aa"

    @staticmethod
    def export_updated_metadata(dataset: RepertoireDataset, result_path: Path, repertoire_folder: str):
        df = pd.read_csv(dataset.metadata_file, comment=Constants.COMMENT_SIGN)
        identifiers = df["identifier"].values.tolist() if "identifier" in df.columns else dataset.get_example_ids()
        df["filename"] = [str(Path(repertoire_folder) / f"{repertoire.data_filename.stem}.tsv") for repertoire in dataset.get_data()]
        df['identifier'] = identifiers
        df.to_csv(result_path / "metadata.csv", index=False)

    @staticmethod
    def _repertoire_to_dataframe(repertoire: Repertoire):
        # get all fields (including custom fields)
        df = pd.DataFrame(repertoire.load_data())

        for column in ['v_alleles', 'j_alleles', 'v_genes', 'j_genes']:
            if column not in df.columns:
                df.loc[:, column] = ''

        AIRRExporter.update_gene_columns(df, 'alleles', 'genes')

        region_type = repertoire.get_region_type()

        # rename mandatory fields for airr-compliance
        mapper = {"sequence_identifiers": "sequence_id", "v_alleles": "v_call", "j_alleles": "j_call", "chains": "locus", "counts": "duplicate_count",
                  "sequences": AIRRExporter.get_sequence_field(region_type), "sequence_aas": AIRRExporter.get_sequence_aa_field(region_type)}

        df = df.rename(mapper=mapper, axis="columns")
        return df

    @staticmethod
    def _receptors_to_dataframe(receptors: List[Receptor]):
        sequences = [(receptor.get_chain(receptor.get_chains()[0]), receptor.get_chain(receptor.get_chains()[1])) for receptor in receptors]
        sequences = [item for sublist in sequences for item in sublist]
        receptor_ids = [(receptor.identifier, receptor.identifier) for receptor in receptors]
        receptor_ids = [item for sublist in receptor_ids for item in sublist]

        df = AIRRExporter._sequences_to_dataframe(sequences)
        df["cell_id"] = receptor_ids
        return df

    @staticmethod
    def _get_sequence_list_region_type(sequences: List[ReceptorSequence]):
        region_types = set([sequence.get_attribute("region_type") for sequence in sequences])

        assert len(region_types) == 1, f"AIRRExporter: expected one region_type, found: {region_types}"

        return RegionType(region_types.pop())

    @staticmethod
    def _sequences_to_dataframe(sequences: List[ReceptorSequence]):
        region_type = AIRRExporter._get_sequence_list_region_type(sequences)
        sequence_field = AIRRExporter.get_sequence_field(region_type)
        sequence_aa_field = AIRRExporter.get_sequence_aa_field(region_type)

        main_data_dict = {"sequence_id": [], sequence_field: [], sequence_aa_field: []}
        attributes_dict = {"chain": [], "v_allele": [], 'v_gene': [], "j_allele": [], 'j_gene': [], "count": [], "cell_id": [], "frame_type": []}

        for i, sequence in enumerate(sequences):
            main_data_dict["sequence_id"].append(sequence.identifier)
            main_data_dict[sequence_field].append(sequence.nucleotide_sequence)
            main_data_dict[sequence_aa_field].append(sequence.amino_acid_sequence)

            # add custom params of this receptor sequence to attributes dict
            if sequence.metadata is not None and sequence.metadata.custom_params is not None:
                for custom_param in sequence.metadata.custom_params:
                    if custom_param not in attributes_dict:
                        attributes_dict[custom_param] = ['' for i in range(i)]

            for attribute in attributes_dict.keys():
                try:
                    attr_value = sequence.get_attribute(attribute)
                    if isinstance(attr_value, Enum):
                        attr_value = attr_value.value
                    attributes_dict[attribute].append(attr_value)
                except KeyError:
                    attributes_dict[attribute].append('')

        df = pd.DataFrame({**attributes_dict, **main_data_dict})

        AIRRExporter.update_gene_columns(df, 'allele', 'gene')
        df.rename(columns={"v_allele": "v_call", "j_allele": "j_call", "chain": "locus", "count": "duplicate_count", "frame_type": "frame_types"},
                  inplace=True)

        return df

    @staticmethod
    def update_gene_columns(df, allele_name, gene_name):
        for index, row in df.iterrows():
            for gene in ['v', 'j']:
                if NumpyHelper.is_nan_or_empty(row[f"{gene}_{allele_name}"]) and not NumpyHelper.is_nan_or_empty(row[f"{gene}_{gene_name}"]):
                    df[f"{gene}_{allele_name}"][index] = row[f"{gene}_{gene_name}"]

    @staticmethod
    def _postprocess_dataframe(df):
        if "locus" in df.columns:
            df["locus"] = [Chain.get_chain(chain).value if chain else '' for chain in df["locus"]]

        if "frame_types" in df.columns:
            AIRRExporter._enums_to_strings(df, "frame_types")

            df["productive"] = df["frame_types"] == SequenceFrameType.IN.name
            df.loc[df["frame_types"].isnull(), "productive"] = ''

            df["vj_in_frame"] = df["productive"]

            df["stop_codon"] = df["frame_types"] == SequenceFrameType.STOP.name
            df.loc[df["frame_types"].isnull(), "stop_codon"] = ''

            df.drop(columns=["frame_types"])

        if "region_types" in df.columns:
            df.drop(columns=["region_types"])

        return df

    @staticmethod
    def _enums_to_strings(df, field):
        df.loc[:, field] = [field_value.value if isinstance(field_value, Enum) else field_value for field_value in df.loc[:, field]]
