import copy
import math
import pickle
from typing import List
from source.data_model.dataset.ElementDataset import ElementDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class SequenceDataset(ElementDataset):

    @classmethod
    def build(cls, sequences: List[ReceptorSequence], file_size: int, path: str, name: str = None):

        file_count = math.ceil(len(sequences) / file_size)
        file_names = [f"{path}batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.pickle"
                      for index in range(1, file_count+1)]

        for index in range(file_count):
            with open(file_names[index], "wb") as file:
                pickle.dump(sequences[index*file_size:(index+1)*file_size], file)

        return SequenceDataset(filenames=file_names, file_size=file_size, name=name)


    def clone(self):
        return SequenceDataset(self.params, copy.deepcopy(self.encoded_data), copy.deepcopy(self._filenames), file_size=self.file_size)
