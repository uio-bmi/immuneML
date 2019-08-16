import math
from collections import Counter
from multiprocessing.pool import Pool

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqRepertoireEncoder(KmerFrequencyEncoder):

    def _encode_new_dataset(self, dataset, params: EncoderParams):

        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = RepertoireDataset(filenames=dataset.get_filenames(),
                                            encoded_data=encoded_data,
                                            params=dataset.params,
                                            metadata_file=dataset.metadata_file)

        self.store(encoded_dataset, params)

        return encoded_dataset

    def _encode_examples(self, dataset, params: EncoderParams):

        arguments = [(filename, dataset, params) for filename in dataset.get_filenames()]

        with Pool(params["batch_size"]) as pool:
            chunksize = math.floor(len(dataset.get_filenames())/params["batch_size"]) + 1
            repertoires = pool.starmap(self._encode_repertoire, arguments, chunksize=chunksize)

        encoded_repertoire_list, repertoire_names, labels, feature_annotation_names = zip(*repertoires)

        encoded_labels = {k: [dic[k] for dic in labels] for k in labels[0]}

        feature_annotation_names = feature_annotation_names[0]

        return list(encoded_repertoire_list), list(repertoire_names), encoded_labels, feature_annotation_names

    def _encode_repertoire(self, filename: str, dataset, params: EncoderParams):

        repertoire = dataset.get_repertoire(filename=filename)
        counts = Counter()
        sequence_encoder = self._prepare_sequence_encoder(params)
        feature_names = sequence_encoder.get_feature_names(params)
        for sequence in repertoire.sequences:
            counts = self._encode_sequence(sequence, params, sequence_encoder, counts)

        label_config = params["label_configuration"]
        labels = dict()

        for label_name in label_config.get_labels_by_name():
            label = repertoire.metadata.custom_params[label_name]
            labels[label_name] = label

        # TODO: refactor this not to return 4 values but e.g. a dict or split into different functions?
        return counts, repertoire.identifier, labels, feature_names
