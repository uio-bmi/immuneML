from collections import Counter

from immuneML.data_model.SequenceParams import Chain
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder


class KmerFreqReceptorEncoder(KmerFrequencyEncoder):

    def _encode_locus(self, dataset):
        return True

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_examples(self, dataset: ReceptorDataset, params: EncoderParams):
        encoded_receptors_counts = []
        receptor_ids = []
        label_config = params.label_config
        labels = {label: [] for label in label_config.get_labels_by_name()} if params.encode_labels else None

        params.region_type = self.region_type
        encode_locus = self._encode_locus(dataset)

        sequence_encoder = self._prepare_sequence_encoder()
        for receptor in dataset.get_data(region_type=self.region_type):
            chains = [Chain.get_chain(chain).name.lower() for chain in receptor.chain_pair.value]
            counts = {chain: Counter() for chain in chains}
            for chain in chains:
                counts[chain] = self._encode_sequence(getattr(receptor, chain), params, sequence_encoder,
                                                      counts[chain], encode_locus)
            encoded_receptors_counts.append(counts[chains[0]] + counts[chains[1]])
            receptor_ids.append(receptor.receptor_id)

            if params.encode_labels:
                for label_name in label_config.get_labels_by_name():
                    label = receptor.metadata[label_name]
                    labels[label_name].append(label)

        return encoded_receptors_counts, receptor_ids, labels
