import numpy as np

from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
from immuneML.util.EncoderHelper import EncoderHelper


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

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_data(self, dataset: ReceptorDataset, params: EncoderParams):
        data = dataset.data

        receptor_ids, chains, mask1, mask2 = EncoderHelper.get_receptor_chain_masks(dataset)

        first_chain_seqs = data[mask1]
        second_chain_seqs = data[mask2]

        sequence_field = self._get_seq_field_name(params)

        max_seq_len = max(getattr(data, sequence_field).lengths)

        labels = (EncoderHelper.encode_element_dataset_labels(dataset, params.label_config, data=data, mask=mask1)
                  if params.encode_labels else None)

        examples_first_chain = self._encode_sequence_list(first_chain_seqs, pad_n_sequences=len(data) // 2,
                                                          pad_sequence_len=max_seq_len, params=params)
        examples_second_chain = self._encode_sequence_list(second_chain_seqs, pad_n_sequences=len(data) // 2,
                                                           pad_sequence_len=max_seq_len, params=params)

        examples = np.stack((examples_first_chain, examples_second_chain), axis=1)

        feature_names = self._get_feature_names(max_seq_len, chains)

        if self.flatten:
            examples = examples.reshape((len(data) // 2, 2*max_seq_len*len(self.onehot_dimensions)))
            feature_names = [item for sublist in feature_names for subsublist in sublist for item in subsublist]

        encoded_data = EncodedData(examples=examples,
                                   labels=labels,
                                   example_ids=receptor_ids,
                                   feature_names=feature_names,
                                   encoding=OneHotEncoder.__name__,
                                   info={"chain_names": chains})

        return encoded_data

    def _get_feature_names(self, max_seq_len, chains):
        return [[[f"{chain}_{pos}_{dim}" for dim in self.onehot_dimensions] for pos in range(max_seq_len)] for chain in chains]
