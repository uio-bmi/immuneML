import pandas as pd

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """
    @staticmethod
    def build(sequences: list, path: str, labels: dict = None, seq_metadata: list = None):
        if seq_metadata is not None:
            assert len(sequences) == len(seq_metadata)
            for index, sequence_list in enumerate(sequences):
                assert len(sequence_list) == len(seq_metadata[index])

        PathBuilder.build(path)

        repertoires = []
        donors = []

        for rep_index, sequence_list in enumerate(sequences):
            rep_sequences = []
            donors.append("rep_" + str(rep_index))
            for seq_index, sequence in enumerate(sequence_list):
                if seq_metadata is None:
                    m = SequenceMetadata(v_gene="v1", j_gene="j1")
                else:
                    m = SequenceMetadata(**seq_metadata[rep_index][seq_index])

                s = ReceptorSequence(amino_acid_sequence=sequence, metadata=m, identifier=str(seq_index))
                rep_sequences.append(s)

            if labels is not None:
                metadata = {key: labels[key][rep_index] for key in labels.keys()}
            else:
                metadata = {}

            repertoire = SequenceRepertoire.build_from_sequence_objects(rep_sequences, path, donors[-1], metadata)
            repertoires.append(repertoire)

        df = pd.DataFrame({**{"filename": [f"{repertoire.identifier}_data.npy" for repertoire in repertoires], "donor": donors},
                           **(labels if labels is not None else {})})
        df.to_csv(path + "metadata.csv", index=False)

        return repertoires, path + "metadata.csv"
