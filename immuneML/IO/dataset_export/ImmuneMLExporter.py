# quality: gold

import copy
import os
import platform
import shutil
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.util.PathBuilder import PathBuilder


class ImmuneMLExporter(DataExporter):

    @staticmethod
    def export(dataset: Dataset, path: Path, number_of_processes: int = 1):
        PathBuilder.build(path)
        exported_dataset = dataset.clone(keep_identifier=True)
        dataset_name = exported_dataset.name
        dataset_filename = f"{dataset_name}.yaml"

        if isinstance(dataset, RepertoireDataset):
            repertoires_path = PathBuilder.build(path / "repertoires")
            exported_repertoires = ImmuneMLExporter._export_repertoires(dataset.repertoires, repertoires_path)
            exported_dataset.repertoires = exported_repertoires
            exported_dataset.metadata_file = ImmuneMLExporter._export_metadata(dataset, path, dataset_filename, repertoires_path)

            file_path = path / dataset_filename
            with file_path.open("w") as file:
                yaml_dict = {
                    **{key: ImmuneMLExporter._parse_val_for_export(val) for key, val in vars(exported_dataset).items()
                       if key not in ['repertoires', 'element_generator', 'encoded_data']},
                    **{'dataset_class': type(exported_dataset).__name__}}
                yaml.dump(yaml_dict, file)

        elif isinstance(dataset, SequenceDataset) or isinstance(dataset, ReceptorDataset):
            exported_dataset.set_filenames(ImmuneMLExporter._export_receptors(exported_dataset.get_filenames(), path))
            exported_dataset.dataset_file = ImmuneMLExporter._export_element_metadata(dataset, path)

        version_path = path / "info.txt"
        with version_path.open("w") as file:
            file.writelines(f"immuneML_version: {Constants.VERSION}\n"
                            f"Python_version: {platform.python_version()}\n")

        return exported_dataset

    @staticmethod
    def _parse_val_for_export(val):
        if isinstance(val, Path) or isinstance(val, Enum):
            return str(val.name)
        elif isinstance(val, list) and all(isinstance(v, Path) for v in val):
            return [str(v.name) for v in val]
        elif isinstance(val, dict):
            return {inner_key: ImmuneMLExporter._parse_val_for_export(inner_val) for inner_key, inner_val in val.items()}
        else:
            return val

    @staticmethod
    def _export_element_metadata(dataset, metadata_folder_path: Path):
        if dataset.dataset_file is None or not dataset.dataset_file.is_file():
            return None

        dataset_file = metadata_folder_path / f"{dataset.name}.yaml"
        if not dataset_file.is_file():
            shutil.copyfile(dataset.dataset_file, dataset_file)

        return dataset_file

    @staticmethod
    def _export_metadata(dataset, metadata_folder_path: Path, dataset_filename, repertoires_path):
        if dataset.metadata_file is None or not dataset.metadata_file.is_file():
            return None

        metadata_file = metadata_folder_path / f"{dataset.name}_metadata.csv"

        if not metadata_file.is_file():
            shutil.copyfile(dataset.metadata_file, metadata_file)

        ImmuneMLExporter._update_repertoire_paths_in_metadata(metadata_file, repertoires_path)
        ImmuneMLExporter._add_dataset_to_metadata(metadata_file, dataset_filename)

        old_metadata_file = metadata_folder_path / "metadata.csv"
        if old_metadata_file.is_file():
            os.remove(str(old_metadata_file))

        return metadata_file

    @staticmethod
    def _update_repertoire_paths_in_metadata(metadata_file: Path, repertoires_path: Path):
        metadata = pd.read_csv(metadata_file, comment=Constants.COMMENT_SIGN)
        path = Path(os.path.relpath(str(repertoires_path), str(metadata_file.parent)))
        metadata["filename"] = [path / os.path.basename(name) for name in metadata["filename"].values.tolist()]
        metadata.to_csv(metadata_file, index=False)

    @staticmethod
    def _add_dataset_to_metadata(metadata_file: Path, dataset_filename: str):
        metadata = pd.read_csv(metadata_file)
        with metadata_file.open("w") as file:
            file.writelines([f"{Constants.COMMENT_SIGN}{dataset_filename}\n"])
        metadata.to_csv(metadata_file, mode="a", index=False)

    @staticmethod
    def _export_receptors(filenames_old: List[Path], path: Path) -> List[Path]:
        filenames_new = []
        for filename_old in filenames_old:
            filename_new = ImmuneMLExporter._copy_if_exists(filename_old, path)
            filenames_new.append(filename_new)
        return filenames_new

    @staticmethod
    def _export_repertoires(repertoires: List[Repertoire], repertoires_path: Path) -> List[Repertoire]:
        new_repertoires = []

        for repertoire_old in repertoires:
            repertoire = copy.deepcopy(repertoire_old)
            repertoire.data_filename = ImmuneMLExporter._copy_if_exists(repertoire_old.data_filename, repertoires_path)
            repertoire.metadata_filename = ImmuneMLExporter._copy_if_exists(repertoire_old.metadata_filename, repertoires_path)
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
            raise RuntimeError(f"{ImmuneMLExporter.__name__}: tried exporting file {old_file}, but it does not exist.")
