from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.EncodedData import EncodedData
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder


class OneHotSequenceEncoder(OneHotEncoder):
    """
    One-hot encoded repertoire data is represented in a matrix with dimensions:
        [sequences, sequence_lengths, one_hot_characters]

    when use_positional_info is true, the last 3 indices in one_hot_characters represents the positional information:
        - start position (high when close to start)
        - middle position (high in the middle of the sequence)
        - end position (high when close to end)
    """

    def _encode_new_dataset(self, dataset: SequenceDataset, params: EncoderParams):
        encoded_data = self._encode_data(dataset, params)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = encoded_data

        return encoded_dataset

    def _encode_data(self, dataset: SequenceDataset, params: EncoderParams):
        data = dataset.data

        sequence_field = self._get_seq_field_name(params)

        max_seq_len = max(getattr(data, sequence_field).lengths)
        labels = self._get_labels(data, params) if params.encode_labels else None

        examples = self._encode_sequence_list(data, pad_n_sequences=len(data),
                                              pad_sequence_len=max_seq_len, params=params)

        feature_names = self._get_feature_names(max_seq_len)

        if self.flatten:
            examples = examples.reshape((len(data), max_seq_len * len(self.onehot_dimensions)))
            feature_names = [item for sublist in feature_names for item in sublist]

        encoded_data = EncodedData(examples=examples,
                                   labels=labels,
                                   example_ids=dataset.get_example_ids(),
                                   feature_names=feature_names,
                                   encoding=OneHotEncoder.__name__)

        return encoded_data

    def _get_feature_names(self, max_seq_len):
        return [[f"{pos}_{dim}" for dim in self.onehot_dimensions] for pos in range(max_seq_len)]

    def _get_labels(self, data: AIRRSequenceSet, params: EncoderParams):
        label_names = params.label_config.get_labels_by_name()
        labels = {name: getattr(data, name).tolist() for name in label_names}

        return labels
