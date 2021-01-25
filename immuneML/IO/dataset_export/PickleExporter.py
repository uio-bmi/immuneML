# quality: gold

import copy
import os
import pickle
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.util.PathBuilder import PathBuilder


class PickleExporter(DataExporter):

    @staticmethod
    def export(dataset: Dataset, path: Path):
        PathBuilder.build(path)
        exported_dataset = copy.deepcopy(dataset)
        dataset_name = exported_dataset.name if exported_dataset.name is not None else exported_dataset.identifier
        dataset_filename = f"{dataset_name}.iml_dataset"

        if isinstance(dataset, RepertoireDataset):
            repertoires_path = PathBuilder.build(path / "repertoires")
            exported_repertoires = PickleExporter._export_repertoires(dataset.repertoires, repertoires_path)
            exported_dataset.repertoires = exported_repertoires
            exported_dataset.metadata_file = PickleExporter._export_metadata(dataset, path, dataset_filename, repertoires_path)
        elif isinstance(dataset, SequenceDataset) or isinstance(dataset, ReceptorDataset):
            exported_dataset.set_filenames(PickleExporter._export_receptors(exported_dataset.get_filenames(), path))

        file_path = path / dataset_filename
        with file_path.open("wb") as file:
            pickle.dump(exported_dataset, file, pickle.HIGHEST_PROTOCOL)

        return exported_dataset

    @staticmethod
    def _export_metadata(dataset, metadata_folder_path: Path, dataset_filename, repertoires_path):
        if dataset.metadata_file is None or not dataset.metadata_file.is_file():
            return None

        metadata_file = metadata_folder_path / f"{dataset.name}_metadata.csv"

        if not metadata_file.is_file():
            shutil.copyfile(dataset.metadata_file, metadata_file)

        PickleExporter._update_repertoire_paths_in_metadata(metadata_file, repertoires_path)
        PickleExporter._add_dataset_to_metadata(metadata_file, dataset_filename)

        old_metadata_file = metadata_folder_path / "metadata.csv"
        if old_metadata_file.is_file():
            os.remove(old_metadata_file)

        return metadata_file

    @staticmethod
    def _update_repertoire_paths_in_metadata(metadata_file: Path, repertoires_path: Path):
        metadata = pd.read_csv(metadata_file, comment=Constants.COMMENT_SIGN)
        path = Path(os.path.relpath(repertoires_path, os.path.dirname(metadata_file)))
        metadata["filename"] = [path / os.path.basename(name) for name in metadata["filename"].values.tolist()]
        metadata.to_csv(metadata_file, index=False)

    @staticmethod
    def _add_dataset_to_metadata(metadata_file: Path, dataset_filename: str):
        metadata = pd.read_csv(metadata_file)
        with metadata_file.open("w") as file:
            file.writelines([f"{Constants.COMMENT_SIGN}{dataset_filename}\n"])
        metadata.to_csv(metadata_file, mode="a", index=False)

    @staticmethod
    def _export_receptors(filenames_old: List[str], path: Path) -> List[str]:
        filenames_new = []
        for filename_old in filenames_old:
            filename_new = PickleExporter._copy_if_exists(filename_old, path)
            filenames_new.append(filename_new)
        return filenames_new

    @staticmethod
    def _export_repertoires(repertoires: List[Repertoire], repertoires_path: Path) -> List[Repertoire]:
        new_repertoires = []

        for repertoire_old in repertoires:
            repertoire = copy.deepcopy(repertoire_old)
            repertoire.data_filename = PickleExporter._copy_if_exists(repertoire_old.data_filename, repertoires_path)
            repertoire.metadata_filename = PickleExporter._copy_if_exists(repertoire_old.metadata_filename, repertoires_path)
            new_repertoires.append(repertoire)

        return new_repertoires

    @staticmethod
    def _copy_if_exists(old_file: Path, path: Path):
        if old_file is not None and old_file.is_file():
            new_file = path / old_file.name
            if not new_file.is_file():
                shutil.copyfile(old_file, new_file)
            return new_file
        else:
            raise RuntimeError(f"{PickleExporter.__name__}: tried exporting file {old_file}, but it does not exist.")
