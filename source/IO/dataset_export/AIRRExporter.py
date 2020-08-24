# quality: gold
import logging

import airr
import pandas as pd

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.util.PathBuilder import PathBuilder


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
    def export(dataset: RepertoireDataset, path, region_type="CDR3"):
        if not isinstance(dataset, RepertoireDataset):
            logging.warning(f"AIRRExporter: dataset {dataset.name} is a {type(dataset).__name__}, but only repertoire dataset export is currently "
                            f"supported for AIRR format.")
        else:
            PathBuilder.build(path)
            repertoire_path = PathBuilder.build(f"{path}repertoires/")

            for repertoire in dataset.repertoires:
                df = AIRRExporter._repertoire_to_dataframe(repertoire, region_type)
                airr.dump_rearrangement(df, f"{repertoire_path}{repertoire.identifier}.tsv")

            AIRRExporter.export_updated_metadata(dataset, path)

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

        if region_type == "CDR3":
            mapper["sequences"] = "junction"
            mapper["sequence_aas"] = "junction_aa"
            if "sequences" in df.columns:
                df["sequences"] = AIRRExporter._process_junctions(df["sequences"])
            if "sequence_aas" in df.columns:
                df["sequence_aas"] = AIRRExporter._process_junction_aas(df["sequence_aas"])
        else:
            mapper["sequences"] = "sequence"
            mapper["sequence_aas"] = "sequence_aa"

        df = df.rename(mapper=mapper, axis="columns")

        if "locus" in df.columns:
            chain_conversion_dict = {Chain.ALPHA: "TRA",
                                     Chain.BETA: "TRB",
                                     Chain.HEAVY: "IGH",
                                     Chain.LIGHT: "IGL"}

            df["locus"] = [chain_conversion_dict[chain] for chain in df["locus"]]

        return df

        # other required fields are: rev_comp, productive, d_call, sequence_alignment
        # germline_alignment, v_cigar, j_cigar, d_cigar

    @staticmethod
    def _process_junctions(column):
        return ["".join(["TG?", value, "T??"]) for value in column]

    @staticmethod
    def _process_junction_aas(column):
        return ["".join(["C", value, "?"]) for value in column]