# quality: gold

from pathlib import Path

from gensim.models import Word2Vec

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.encodings.word2vec.model_creator.ModelCreatorStrategy import ModelCreatorStrategy
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.KmerHelper import KmerHelper


class SequenceModelCreator(ModelCreatorStrategy):

    def create_model(self, dataset: Dataset, k: int, vector_size: int, batch_size: int, model_path: Path, sequence_type: SequenceType):
        model = Word2Vec(vector_size=vector_size, min_count=1, window=self.window)  # creates an empty model
        all_kmers = KmerHelper.create_all_kmers(k=k, alphabet=EnvironmentSettings.get_sequence_alphabet())
        all_kmers = [[kmer] for kmer in all_kmers]
        model.build_vocab(all_kmers)

        if isinstance(dataset, RepertoireDataset):
            model = self._create_for_repertoire(dataset, batch_size, k, model, all_kmers, sequence_type)
        elif isinstance(dataset, SequenceDataset):
            model = self._create_for_sequences(dataset, batch_size, k, model, all_kmers, sequence_type)

        model.save(str(model_path))

        return model

    def _create_for_repertoire(self, dataset, batch_size, k, model, all_kmers, sequence_type):
        for example in dataset.get_data(batch_size=batch_size):
            sentences = KmerHelper.create_sentences_from_repertoire(repertoire=example, k=k, sequence_type=sequence_type)
            model.train(corpus_iterable=sentences, total_words=len(all_kmers), epochs=self.epochs)
        return model

    def _create_for_sequences(self, dataset: SequenceDataset, batch_size, k, model, all_kmers, sequence_type):
        for sequence_batch in dataset.get_batch(batch_size):
            sentences = [KmerHelper.create_kmers_from_sequence(seq, k, sequence_type) for seq in sequence_batch]
            model.train(corpus_iterable=sentences, total_words=len(all_kmers), epochs=self.epochs)
        return model
