import pandas as pd

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.ReceptorSequenceList import ReceptorSequenceList
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """
    @staticmethod
    def build(sequences: list, path: str, labels: dict = None, seq_metadata: list = None, donors: list = None):

        if donors is not None:
            assert len(donors) == len(sequences)

        if seq_metadata is not None:
            assert len(sequences) == len(seq_metadata)
            for index, sequence_list in enumerate(sequences):
                assert len(sequence_list) == len(seq_metadata[index])

        PathBuilder.build(path)
        rep_path = PathBuilder.build(path + "repertoires/")

        repertoires = []
        if donors is None:
            donors = []

        for rep_index, sequence_list in enumerate(sequences):
            rep_sequences = ReceptorSequenceList()
            if len(donors) < len(sequences):
                donors.append("rep_" + str(rep_index))
            for seq_index, sequence in enumerate(sequence_list):
                if seq_metadata is None:
                    m = SequenceMetadata(v_gene="v1", j_gene="j1", count=1)
                else:
                    m = SequenceMetadata(**seq_metadata[rep_index][seq_index])

                s = ReceptorSequence(amino_acid_sequence=sequence, metadata=m, identifier=str(seq_index))
                rep_sequences.append(s)

            if labels is not None:
                metadata = {key: labels[key][rep_index] for key in labels.keys()}
            else:
                metadata = {}

            metadata = {**metadata, **{"donor": donors[rep_index]}}

            repertoire = Repertoire.build_from_sequence_objects(rep_sequences, rep_path, metadata)
            repertoires.append(repertoire)

        df = pd.DataFrame({**{"filename": [f"{repertoire.identifier}_data.npy" for repertoire in repertoires], "donor": donors,
                              "repertoire_identifier": [repertoire.identifier for repertoire in repertoires]},
                           **(labels if labels is not None else {})})
        df.to_csv(path + "metadata.csv", index=False)

        return repertoires, path + "metadata.csv"
