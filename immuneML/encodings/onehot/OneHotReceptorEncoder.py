import numpy as np

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder


class OneHotReceptorEncoder(OneHotEncoder):
    """
    One-hot encoded repertoire data is represented in a matrix with dimensions:
        [receptors, chains, sequence_lengths, one_hot_characters]

    when use_positional_info is true, the last 3 indices in one_hot_characters represents the positional information:
        - start position (high when close to start)
        - middle position (high in the middle of the sequence)
        - end position (high when close to end)
    """

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = ReceptorDataset(filenames=dataset.get_filenames(),
                                          encoded_data=encoded_data,
                                          labels=dataset.labels)

        return encoded_dataset

    def _encode_data(self, dataset: ReceptorDataset, params: EncoderParams):
        receptor_objs = [receptor for receptor in dataset.get_data()]
        sequences = [[getattr(obj, chain).get_sequence() for chain in obj.get_chains()] for obj in receptor_objs]
        first_chain_seqs, second_chain_seqs = zip(*sequences)

        max_seq_len = max(max([len(seq) for seq in first_chain_seqs]), max([len(seq) for seq in second_chain_seqs]))

        example_ids = dataset.get_example_ids()
        labels = self._get_labels(receptor_objs, params) if params.encode_labels else None

        examples_first_chain = self._encode_sequence_list(first_chain_seqs, pad_n_sequences=len(receptor_objs),
                                                          pad_sequence_len=max_seq_len)
        examples_second_chain = self._encode_sequence_list(second_chain_seqs, pad_n_sequences=len(receptor_objs),
                                                           pad_sequence_len=max_seq_len)

        examples = np.stack((examples_first_chain, examples_second_chain), axis=1)

        feature_names = self._get_feature_names(max_seq_len, receptor_objs[0].get_chains())

        if self.flatten:
            examples = examples.reshape((len(receptor_objs), 2*max_seq_len*len(self.onehot_dimensions)))
            feature_names = [item for sublist in feature_names for subsublist in sublist for item in subsublist]

        encoded_data = EncodedData(examples=examples,
                                   labels=labels,
                                   example_ids=example_ids,
                                   feature_names=feature_names,
                                   encoding=OneHotEncoder.__name__,
                                   info={"chain_names": receptor_objs[0].get_chains() if all(receptor_obj.get_chains() == receptor_objs[0].get_chains() for receptor_obj in receptor_objs) else None})

        return encoded_data

    def _get_feature_names(self, max_seq_len, chains):
        return [[[f"{chain}_{pos}_{dim}" for dim in self.onehot_dimensions] for pos in range(max_seq_len)] for chain in chains]

    def _get_labels(self, receptor_objs, params: EncoderParams):
        label_names = params.label_config.get_labels_by_name()
        labels = {name: [None] * len(receptor_objs) for name in label_names}

        for idx, receptor in enumerate(receptor_objs):
            for name in label_names:
                labels[name][idx] = receptor.metadata[name]

        return labels
