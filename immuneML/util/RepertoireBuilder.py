from pathlib import Path

import pandas as pd

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """
    @staticmethod
    def build(sequences: list, path: Path, labels: dict = None, seq_metadata: list = None, subject_ids: list = None):

        if subject_ids is not None:
            assert len(subject_ids) == len(sequences)

        if seq_metadata is not None:
            assert len(sequences) == len(seq_metadata)
            for index, sequence_list in enumerate(sequences):
                assert len(sequence_list) == len(seq_metadata[index])

        PathBuilder.build(path)
        rep_path = PathBuilder.build(path / "repertoires")

        repertoires = []
        if subject_ids is None:
            subject_ids = []

        for rep_index, sequence_list in enumerate(sequences):
            rep_sequences = ReceptorSequenceList()
            if len(subject_ids) < len(sequences):
                subject_ids.append("rep_" + str(rep_index))
            for seq_index, sequence in enumerate(sequence_list):
                if seq_metadata is None:
                    m = SequenceMetadata(v_subgroup="TRBV1", v_gene="TRBV1-1", v_allele="TRBV1-1*01", j_subgroup="TRBJ1", j_gene="TRBJ1-1", j_allele="TRBJ1-1*01", count=1, chain="TRB", region_type="IMGT_CDR3")
                else:
                    m = SequenceMetadata(**seq_metadata[rep_index][seq_index])

                s = ReceptorSequence(amino_acid_sequence=sequence, metadata=m, identifier=str(seq_index))
                rep_sequences.append(s)

            if labels is not None:
                metadata = {key: labels[key][rep_index] for key in labels.keys()}
            else:
                metadata = {}

            metadata = {**metadata, **{"subject_id": subject_ids[rep_index]}}

            repertoire = Repertoire.build_from_sequence_objects(rep_sequences, rep_path, metadata)
            repertoires.append(repertoire)

        df = pd.DataFrame({**{"filename": [f"{repertoire.identifier}.npy" for repertoire in repertoires], "subject_id": subject_ids,
                              "repertoire_identifier": [repertoire.identifier for repertoire in repertoires]},
                           **(labels if labels is not None else {})})
        df.to_csv(path / "metadata.csv", index=False)

        return repertoires, path / "metadata.csv"
