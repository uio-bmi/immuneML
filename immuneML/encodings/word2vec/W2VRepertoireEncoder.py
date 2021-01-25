import numpy as np

from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.util.KmerHelper import KmerHelper


class W2VRepertoireEncoder(Word2VecEncoder):

    def _encode_examples(self, encoded_dataset, vectors, params):
        repertoires = np.zeros(shape=[encoded_dataset.get_example_count(), vectors.vector_size])
        for (index, repertoire) in enumerate(encoded_dataset.get_data()):
            repertoires[index] = self._encode_repertoire(repertoire, vectors)
        return repertoires

    def _encode_labels(self, dataset, params: EncoderParams):

        label_config = params.label_config
        labels = {name: [] for name in label_config.get_labels_by_name()}

        for repertoire in dataset.get_data():

            for label_name in label_config.get_labels_by_name():
                label = repertoire.metadata[label_name]
                labels[label_name].append(label)

        return np.array([labels[name] for name in labels.keys()])

    def _encode_repertoire(self, repertoire, vectors):
        repertoire_vector = np.zeros(vectors.vector_size)
        for (index2, sequence) in enumerate(repertoire.sequences):
            kmers = KmerHelper.create_kmers_from_sequence(sequence=sequence, k=self.k)
            sequence_vector = np.zeros(vectors.vector_size)
            for kmer in kmers:
                try:
                    word_vector = vectors.get_vector(kmer)
                    sequence_vector = np.add(sequence_vector, word_vector)
                except KeyError:
                    pass

            repertoire_vector = np.add(repertoire_vector, sequence_vector)
        return repertoire_vector
