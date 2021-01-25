# quality: gold

from pathlib import Path

from gensim.models import Word2Vec

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.word2vec.model_creator.ModelCreatorStrategy import ModelCreatorStrategy
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.KmerHelper import KmerHelper


class SequenceModelCreator(ModelCreatorStrategy):

    def create_model(self, dataset: RepertoireDataset, k: int, vector_size: int, batch_size: int, model_path: Path):
        model = Word2Vec(size=vector_size, min_count=1, window=5)  # creates an empty model
        all_kmers = KmerHelper.create_all_kmers(k=k, alphabet=EnvironmentSettings.get_sequence_alphabet())
        all_kmers = [[kmer] for kmer in all_kmers]
        model.build_vocab(all_kmers)

        for repertoire in dataset.get_data(batch_size=batch_size):
            sentences = KmerHelper.create_sentences_from_repertoire(repertoire=repertoire, k=k)
            model.train(sentences=sentences, total_words=len(all_kmers), epochs=15)

        model.save(str(model_path))

        return model
