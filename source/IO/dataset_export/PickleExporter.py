# quality: gold

import copy
import os
import pickle
import shutil
from typing import List

import pandas as pd

from source.IO.dataset_export.DataExporter import DataExporter
from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.util.PathBuilder import PathBuilder


class PickleExporter(DataExporter):

    @staticmethod
    def export(dataset: Dataset, path):
        PathBuilder.build(path)
        exported_dataset = copy.deepcopy(dataset)
        dataset_name = exported_dataset.name if exported_dataset.name is not None else exported_dataset.identifier
        dataset_filename = f"{dataset_name}.iml_dataset"

        if isinstance(dataset, RepertoireDataset):
            repertoires_path = PathBuilder.build(f"{path}repertoires/")
            exported_repertoires = PickleExporter._export_repertoires(dataset.repertoires, repertoires_path)
            exported_dataset.repertoires = exported_repertoires
            exported_dataset.metadata_file = PickleExporter._export_metadata(dataset, path, dataset_filename, repertoires_path)
        elif isinstance(dataset, SequenceDataset) or isinstance(dataset, ReceptorDataset):
            exported_dataset.set_filenames(PickleExporter._export_receptors(exported_dataset.get_filenames(), path))

        with open(f"{path}/{dataset_filename}", "wb") as file:
            pickle.dump(exported_dataset, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _export_metadata(dataset, metadata_folder_path, dataset_filename, repertoires_path):
        if dataset.metadata_file is None or not os.path.isfile(dataset.metadata_file):
            return None

        metadata_file = f"{metadata_folder_path}{os.path.basename(dataset.metadata_file)}"

        if not os.path.isfile(metadata_file):
            shutil.copyfile(dataset.metadata_file, metadata_file)

        PickleExporter._update_repertoire_paths_in_metadata(metadata_file, repertoires_path)
        PickleExporter._add_dataset_to_metadata(metadata_file, dataset_filename)

        return metadata_file

    @staticmethod
    def _update_repertoire_paths_in_metadata(metadata_file, repertoires_path):
        metadata = pd.read_csv(metadata_file, comment=Constants.COMMENT_SIGN)
        path = os.path.relpath(repertoires_path, os.path.dirname(metadata_file))
        metadata["filename"] = [f"{path}/{os.path.basename(name)}" for name in metadata["filename"].values.tolist()]
        metadata.to_csv(metadata_file, index=False)

    @staticmethod
    def _add_dataset_to_metadata(metadata_file, dataset_filename):
        metadata = pd.read_csv(metadata_file)
        with open(metadata_file, "w") as file:
            file.writelines([f"{Constants.COMMENT_SIGN}{dataset_filename}\n"])
        metadata.to_csv(metadata_file, mode="a", index=False)

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
            repertoire.data_filename = PickleExporter._copy_if_exists(repertoire_old.data_filename, repertoires_path)
            repertoire.metadata_filename = PickleExporter._copy_if_exists(repertoire_old.metadata_filename, repertoires_path)
            new_repertoires.append(repertoire)

        return new_repertoires

    @staticmethod
    def _copy_if_exists(old_file, path):
        if old_file is not None and os.path.isfile(old_file):
            new_file = f"{path}{os.path.basename(old_file)}"
            if not os.path.isfile(new_file):
                shutil.copyfile(old_file, new_file)
        else:
            new_file = None
        return new_file
