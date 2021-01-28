import copy
import math
import pickle
from pathlib import Path
from typing import List

from immuneML.data_model.dataset.ElementDataset import ElementDataset
from immuneML.data_model.receptor.Receptor import Receptor


class ReceptorDataset(ElementDataset):

    @classmethod
    def build(cls, receptors: List[Receptor], file_size: int, path: Path, name: str = None):

        file_count = math.ceil(len(receptors) / file_size)
        file_names = [path / f"batch{''.join(['0' for i in range(1, len(str(file_count)) - len(str(index)) + 1)])}{index}.pickle"
                      for index in range(1, file_count+1)]

        for index in range(file_count):
            with file_names[index].open("wb") as file:
                pickle.dump(receptors[index*file_size:(index+1)*file_size], file)

        return ReceptorDataset(filenames=file_names, file_size=file_size, name=name)

    def clone(self):
        return ReceptorDataset(self.labels, copy.deepcopy(self.encoded_data), copy.deepcopy(self._filenames), file_size=self.file_size)
