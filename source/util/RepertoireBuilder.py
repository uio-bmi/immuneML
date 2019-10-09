import pickle

import pandas as pd

from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """
    @staticmethod
    def build(sequences: list, path: str, labels: dict = None):

        PathBuilder.build(path)

        filenames = []
        donors = []

        for index, sequence_list in enumerate(sequences):
            rep_sequences = []
            for sequence in sequence_list:
                s = ReceptorSequence(amino_acid_sequence=sequence, metadata=SequenceMetadata(v_gene="v1", j_gene="j1"))
                rep_sequences.append(s)
            repertoire = SequenceRepertoire(sequences=rep_sequences, identifier=str(index))

            if labels is not None:
                rep_labels = {key: labels[key][index] for key in labels.keys()}
                repertoire.metadata = RepertoireMetadata(custom_params=rep_labels)

            filenames.append(path + "{}.pkl".format(index))
            donors.append("rep_" + str(index))

            with open(filenames[-1], "wb") as file:
                pickle.dump(repertoire, file)

        df = pd.DataFrame({**{"filename": filenames, "donor": donors}, **(labels if labels is not None else {})})
        df.to_csv(path + "metadata.csv", index=False)

        return filenames, path + "metadata.csv"
