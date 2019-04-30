import pickle

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """
    @staticmethod
    def build(sequences: list, path: str, labels: dict = None):

        PathBuilder.build(path)

        filenames = []

        for index, sequence_list in enumerate(sequences):
            rep_sequences = []
            for sequence in sequence_list:
                s = ReceptorSequence(amino_acid_sequence=sequence, metadata=SequenceMetadata())
                rep_sequences.append(s)
            repertoire = Repertoire(sequences=rep_sequences, identifier=str(index))

            if labels is not None:
                rep_labels = {key: labels[key][index] for key in labels.keys()}
                repertoire.metadata = RepertoireMetadata(custom_params=rep_labels)

            filenames.append(path + "{}.pkl".format(index))

            with open(filenames[-1], "wb") as file:
                pickle.dump(repertoire, file)

        return filenames
