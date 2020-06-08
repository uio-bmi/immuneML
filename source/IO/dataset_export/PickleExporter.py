# quality: gold

import copy
import os
import pickle
import shutil
from typing import List

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.util.PathBuilder import PathBuilder


class PickleExporter(DataExporter):

    @staticmethod
    def export(dataset: Dataset, path):
        PathBuilder.build(path)
        exported_dataset = copy.deepcopy(dataset)

        if isinstance(dataset, RepertoireDataset):
            repertoires_path = PathBuilder.build(f"{path}repertoires/")
            exported_repertoires = PickleExporter._export_repertoires(dataset.repertoires, repertoires_path)
            exported_dataset.repertoires = exported_repertoires
        elif isinstance(dataset, SequenceDataset) or isinstance(dataset, ReceptorDataset):
            exported_dataset.set_filenames(PickleExporter._export_receptors(exported_dataset.get_filenames(), path))

        dataset_name = exported_dataset.name if exported_dataset.name is not None else exported_dataset.identifier

        with open(f"{path}/{dataset_name}.iml_dataset", "wb") as file:
            pickle.dump(exported_dataset, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _export_receptors(filenames_old: List[str], path: str) -> List[str]:
        filenames_new = []
        for filename_old in filenames_old:
            filename_new = f"{path}{os.path.basename(filename_old)}"
            shutil.copyfile(filename_old, filename_new)
            filenames_new.append(filename_new)
        return filenames_new

    @staticmethod
    def _export_repertoires(repertoires: List[Repertoire], repertoires_path: str) -> List[Repertoire]:
        new_repertoires = []

        for repertoire_old in repertoires:
            repertoire = copy.deepcopy(repertoire_old)

            if repertoire_old.data_filename is not None and os.path.isfile(repertoire_old.data_filename):
                repertoire.data_filename = f"{repertoires_path}{os.path.basename(repertoire_old.data_filename)}"
                shutil.copyfile(repertoire_old.data_filename, repertoire.data_filename)
            else:
                repertoire.data_filename = None

            if repertoire_old.metadata_filename is not None and os.path.isfile(repertoire_old.metadata_filename):
                repertoire.metadata_filename = f"{repertoires_path}{os.path.basename(repertoire_old.metadata_filename)}"
                shutil.copyfile(repertoire_old.metadata_filename, repertoire.metadata_filename)
            else:
                repertoire.metadata_filename = None

            new_repertoires.append(repertoire)

        return new_repertoires
