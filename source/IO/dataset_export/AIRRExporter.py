# quality: gold

import airr
import pandas as pd

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.PathBuilder import PathBuilder


class AIRRExporter(DataExporter):
    '''
    Exports a RepertoireDataset of SequenceRepertoires in AIRR format.

    Things to note:
        - one filename_prefix is given, which is combined with the Repertoire identifiers
        for the filenames, to create one file per Repertoire
        - 'counts' is written into the field 'duplicate_counts'
        - 'sequence_identifiers' is written both into the fields 'sequence_id' and 'rearrangement_id'
    '''

    @staticmethod
    def export(dataset: RepertoireDataset, path, filename_prefix):
        PathBuilder.build(path)

        for repertoire in dataset.repertoires:
            df = AIRRExporter._repertoire_to_dataframe(repertoire)
            airr.dump_rearrangement(df, path + f"{filename_prefix}_{repertoire.identifier}.tsv")

    @staticmethod
    def _repertoire_to_dataframe(repertoire: Repertoire):
        # get all fields (including custom fields)
        df = pd.DataFrame({key: repertoire.get_attribute(key) for key in set(repertoire._fields)})

        # rename mandatory fields for airr-compliancy
        df = df.rename(mapper = {"sequences": "sequence",
                            "sequence_aas": "sequence_aa",
                            "sequence_identifiers": "rearrangement_id",
                            "v_genes": "v_call",
                            "j_genes": "j_call",
                            "chains": "locus",
                            "counts": "duplicate_count"}, axis = "columns")

        df["sequence_id"] = df["rearrangement_id"]

        if "locus" in df.columns:
            chain_conversion_dict = {Chain.A: "TRA",
                                     Chain.B: "TRB",
                                     Chain.H: "IGH",
                                     Chain.L: "IGL"}

            df["locus"] = [chain_conversion_dict[chain] for chain in df["locus"]]

        return df

        # other required fields are: rev_comp, productive, d_call, sequence_alignment
        # germline_alignment, junction, junction_aa, v_cigar, j_cigar, d_cigar

