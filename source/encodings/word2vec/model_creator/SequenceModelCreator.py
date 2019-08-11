# quality: gold

from gensim.models import Word2Vec

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.word2vec.model_creator.ModelCreatorStrategy import ModelCreatorStrategy
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.KmerHelper import KmerHelper


class SequenceModelCreator(ModelCreatorStrategy):

    def create_model(self, dataset: RepertoireDataset, params: EncoderParams, model_path):
        k = params["model"]["k"]

        model = Word2Vec(size=params["model"]["size"], min_count=1, window=5)  # creates an empty model
        all_kmers = KmerHelper.create_all_kmers(k=k, alphabet=EnvironmentSettings.get_sequence_alphabet())
        all_kmers = [[kmer] for kmer in all_kmers]
        model.build_vocab(all_kmers)

        for repertoire in dataset.get_data(batch_size=params["batch_size"]):
            sentences = KmerHelper.create_sentences_from_repertoire(repertoire=repertoire, k=k)
            model.train(sentences=sentences, total_words=len(all_kmers), epochs=15)

        model.save(model_path)

        return model
