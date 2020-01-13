import copy

from source.data_model.dataset.ElementDataset import ElementDataset


class SequenceDataset(ElementDataset):

    def clone(self):
        return SequenceDataset(self.params, copy.deepcopy(self.encoded_data), copy.deepcopy(self._filenames), file_size=self.file_size)
