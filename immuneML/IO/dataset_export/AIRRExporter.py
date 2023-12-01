import logging
import math
from dataclasses import fields
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import List

import airr
import pandas as pd
from olga.utils import nt2aa

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
    def export(dataset: Dataset, path: Path, number_of_processes: int = 1, omit_columns: list = None):
        PathBuilder.build(path)

        if isinstance(dataset, RepertoireDataset):
            repertoire_folder = "repertoires/"
            repertoire_path = PathBuilder.build(path / repertoire_folder)

            with Pool(processes=number_of_processes) as pool:
                arguments = [(repertoire, repertoire_path, dataset.labels, omit_columns)
                             for repertoire in dataset.repertoires]
                pool.starmap(AIRRExporter.export_repertoire, arguments)

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

                df = AIRRExporter._postprocess_dataframe(df, dataset.labels, omit_columns)
                airr.dump_rearrangement(df, str(filename))

                index += 1

    @staticmethod
    def export_repertoire(repertoire: Repertoire, repertoire_path: Path, dataset_labels: dict, omit_columns: list = None):
        df = AIRRExporter._repertoire_to_dataframe(repertoire)
        df = AIRRExporter._postprocess_dataframe(df, dataset_labels, omit_columns)
        output_file = repertoire_path / f"{repertoire.data_filename.stem if 'subject_id' not in repertoire.metadata else repertoire.metadata['subject_id']}.tsv"
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
        df["filename"] = [f"{repertoire.data_filename.stem if 'subject_id' not in repertoire.metadata else repertoire.metadata['subject_id']}.tsv"
                          for repertoire in dataset.get_data()]
        df['identifier'] = identifiers
        df.to_csv(result_path / "metadata.csv", index=False)

    @staticmethod
    def _repertoire_to_dataframe(repertoire: Repertoire):
        rep_data = repertoire.load_bnp_data()
        df = pd.DataFrame({field.name: getattr(rep_data, field.name).tolist() for field in fields(rep_data)})

        region_type = repertoire.get_region_type()

        # rename mandatory fields for airr-compliance
        mapper = {"chain": "locus", "sequence": AIRRExporter.get_sequence_field(region_type),
                  "sequence_aa": AIRRExporter.get_sequence_aa_field(region_type)}

        df = df.rename(mapper=mapper, axis="columns")
        df.drop(columns=['region_type'], inplace=True)

        return df

    @staticmethod
    def add_full_length_seq(df, species, unique_chains):
        if unique_chains is not None and len(unique_chains) <= 2 and all(chain in [Chain.ALPHA.value, Chain.BETA.value] for chain in unique_chains):
            try:
                from Stitchr import stitchr as st
                from Stitchr import stitchrfunctions as fxn

                tcr_dat, functionality, partial = {}, {}, {}

                for chain in unique_chains:
                    tcr_dat[chain], functionality[chain], partial[chain] = fxn.get_imgt_data(chain, st.gene_types, species.upper())

                codons = fxn.get_optimal_codons('', species)

                df['full_sequence'] = df.apply(lambda row: stitch_wrapper(row, st, fxn, species, tcr_dat, functionality, partial, codons), axis=1)

                df['full_sequence_aa'] = df.apply(lambda row: nt2aa(row['full_sequence']), axis=1)

            except Exception as e:
                logging.warning(f"An error occurred while exporting full length sequence. Only CDR3/JUNCTION region "
                                f"is exported instead.\nFull error: {e}")

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
        attributes_dict = {"chain": [], "v_call": [], "j_call": [], "duplicate_count": [], "cell_id": [], "frame_type": []}

        for i, sequence in enumerate(sequences):
            main_data_dict["sequence_id"].append(sequence.sequence_id)
            main_data_dict[sequence_field].append(sequence.sequence)
            main_data_dict[sequence_aa_field].append(sequence.sequence_aa)

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

        df.rename(columns={"chain": "locus"}, inplace=True)

        return df

    @staticmethod
    def update_gene_columns(df, allele_name, gene_name):
        for index, row in df.iterrows():
            for gene in ['v', 'j']:
                if NumpyHelper.is_nan_or_empty(row[f"{gene}_{allele_name}"]) and not NumpyHelper.is_nan_or_empty(row[f"{gene}_{gene_name}"]):
                    df.at[index, f"{gene}_{allele_name}"] = row[f"{gene}_{gene_name}"]

    @staticmethod
    def _postprocess_dataframe(df, dataset_labels: dict, omit_columns: list = None):
        if "locus" in df.columns:
            df["locus"] = [Chain.get_chain(chain).value if chain and Chain.get_chain(chain) else '' for chain in df["locus"]]
        else:
            df['locus'] = df.apply(lambda row: Chain.get_chain(row['v_call'][:3]).value, axis=1)

        if "frame_type" in df.columns:
            AIRRExporter._enums_to_strings(df, "frame_type")

            df["productive"] = df["frame_type"] == SequenceFrameType.IN.value
            df.loc[df["frame_type"].isnull(), "productive"] = ""
            df.loc[df["frame_type"] == "", "productive"] = ""
            df.loc[df["frame_type"] == SequenceFrameType.UNDEFINED.value, "productive"] = ""

            df["vj_in_frame"] = df["productive"]

            df["stop_codon"] = df["frame_type"] == SequenceFrameType.STOP.value
            df.loc[df["frame_type"].isnull(), "stop_codon"] = ''

            df.drop(columns=["frame_type"], inplace=True)

        if "region_type" in df.columns:
            df.drop(columns=["region_type"], inplace=True)

        if omit_columns is not None:
            df.drop(columns=omit_columns, inplace=True)

        AIRRExporter.add_full_length_seq(df, dataset_labels.get('species', None) if dataset_labels else None, list(set(df['locus'].values.tolist())))

        return df

    @staticmethod
    def _enums_to_strings(df, field):
        df.loc[:, field] = [field_value.value if isinstance(field_value, Enum) else field_value for field_value in df.loc[:, field]]


def stitch_wrapper(row, st, fxn, species, tcr_dat, functionality, partial, codons):
    full_sequence = ""

    try:
        full_sequence = st.stitch({'v': row['v_call'], 'j': row['j_call'], 'cdr3': row['junction_aa'],
                   'skip_c_checks': False, '5_prime_seq': '', '3_prime_seq': '', 'name': '',
                   'c': fxn.autofill_input({'c': None, 'species': species.upper(), 'j': row['j_call'],
                                            'l': row['v_call']}, row['locus'])['c'],
                   'species': species.upper(), 'l': row['v_call']},
                  tcr_dat[row['locus']], functionality[row['locus']], partial[row['locus']], codons, 3, '')[1]

    except Exception as e:
        logging.warning(f"An error occurred while constructing full sequence from row: \n{row}. Error log: \n{e}")

    return full_sequence
