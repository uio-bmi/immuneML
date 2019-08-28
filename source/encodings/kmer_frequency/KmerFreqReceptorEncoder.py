from collections import Counter

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqReceptorEncoder(KmerFrequencyEncoder):
    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = ReceptorDataset(filenames=dataset.get_filenames(),
                                          encoded_data=encoded_data,
                                          params=dataset.params)

        self.store(encoded_dataset, params)

        return encoded_dataset

    def _encode_examples(self, dataset, params: EncoderParams):
        encoded_receptors = []
        receptor_ids = []
        label_config = params["label_configuration"]
        labels = {label: [] for label in label_config.get_labels_by_name()}

        sequence_encoder = self._prepare_sequence_encoder(params)
        feature_names = sequence_encoder.get_feature_names(params)
        for receptor in dataset.get_data(params["batch_size"]):
            counts = Counter()
            for chain in receptor.get_chains():
                counts = self._encode_sequence(receptor.get_chain(chain), params, sequence_encoder, counts)
            encoded_receptors.append(counts)
            receptor_ids.append(receptor.identifier)

            for label_name in label_config.get_labels_by_name():
                label = receptor.metadata[label_name]
                labels[label_name].append(label)

        return encoded_receptors, receptor_ids, labels, feature_names
