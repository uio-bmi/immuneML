import pickle
import random

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class RandomDatasetGenerator:

    @staticmethod
    def generate_repertoire(alphabet, sequence_count, sequence_length) -> SequenceRepertoire:
        sequences = []
        for j in range(sequence_count):
            s = "".join(random.choices(alphabet, k=sequence_length))
            sequence = ReceptorSequence(amino_acid_sequence=s)
            sequences.append(sequence)
        return SequenceRepertoire(sequences=sequences)

    @staticmethod
    def store_repertoire(repertoire: SequenceRepertoire, index: int, path):
        filepath = path + "rep" + str(index) + ".pkl"

        with open(filepath, "wb") as file:
            pickle.dump(repertoire, file)

        return filepath

    @staticmethod
    def generate_dataset(repertoire_count: int, sequence_count: int, path: str) -> RepertoireDataset:

        alphabet = EnvironmentSettings.get_sequence_alphabet()
        sequence_length = 12
        filepaths = []
        PathBuilder.build(path)

        for i in range(repertoire_count):
            repertoire = RandomDatasetGenerator.generate_repertoire(alphabet, sequence_count, sequence_length)
            filepath = RandomDatasetGenerator.store_repertoire(repertoire, i, path)
            filepaths.append(filepath)

        return RepertoireDataset(filenames=filepaths)
