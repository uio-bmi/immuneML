
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
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

        encoded_dataset = SequenceDataset(filenames=dataset.get_filenames(),
                                          encoded_data=encoded_data,
                                          labels=dataset.labels,
                                          file_size=dataset.file_size)

        return encoded_dataset

    def _encode_data(self, dataset: SequenceDataset, params: EncoderParams):
        sequence_objs = [obj for obj in dataset.get_data(params.pool_size)]

        sequences = [obj.get_sequence() for obj in sequence_objs]
        example_ids = dataset.get_example_ids()
        max_seq_len = max([len(seq) for seq in sequences])
        labels = self._get_labels(sequence_objs, params) if params.encode_labels else None

        examples = self._encode_sequence_list(sequences, pad_n_sequences=len(sequence_objs), pad_sequence_len=max_seq_len)

        feature_names = self._get_feature_names(max_seq_len)

        if self.flatten:
            examples = examples.reshape((len(sequence_objs), max_seq_len*len(self.onehot_dimensions)))
            feature_names = [item for sublist in feature_names for item in sublist]

        encoded_data = EncodedData(examples=examples,
                                   labels=labels,
                                   example_ids=example_ids,
                                   feature_names=feature_names,
                                   encoding=OneHotEncoder.__name__)

        return encoded_data

    def _get_feature_names(self, max_seq_len):
        return [[f"{pos}_{dim}" for dim in self.onehot_dimensions] for pos in range(max_seq_len)]


    def _get_labels(self, sequence_objs, params: EncoderParams):
        label_names = params.label_config.get_labels_by_name()
        labels = {name: [None] * len(sequence_objs) for name in label_names}

        for idx, sequence in enumerate(sequence_objs):
            for name in label_names:
                labels[name][idx] = sequence.get_attribute(name)

        return labels


