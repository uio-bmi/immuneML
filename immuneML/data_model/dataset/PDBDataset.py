import os
from pathlib import Path
from uuid import uuid4
import pandas as pd
import copy

from immuneML.data_model.receptor.ElementGenerator import ElementGenerator

from immuneML.data_model.pdb_structure.PDBStructure import PDBStructure
from immuneML.environment.Constants import Constants
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.receptor.RegionType import RegionType
from Bio.PDB import *
from typing import List



class PDBDataset(Dataset):
    """
    This is the Dataset class that is used to store the information in a PDB file.
    """


    def __init__(self, pdb_file_paths: List= None, file_names: list = None, labels: dict = None, metadata_file: Path = None, encoded_data: EncodedData = None, identifier: str = None,
                 name: str = None, region_type: RegionType = RegionType.FULL_SEQUENCE, list_of_PDB_Structures: list = None, element_ids: list = None):
        super().__init__()
        self.encoded_data = encoded_data
        self.labels = labels
        self.identifier = identifier if identifier is not None else uuid4().hex
        self.name = name if name is not None else self.identifier
        self.pdb_file_paths = pdb_file_paths
        self.file_names = file_names
        self.metadata_file = metadata_file
        self.region_type = region_type
        self.list_of_PDB_Structures = self.generate_PDB_Structures()




    def check_if_PDB_file_has_IMGT_numbering(self, checked_pdb_filepath):
        file = open(checked_pdb_filepath, "r")

        if "removed_antigen.pdb" in checked_pdb_filepath:
            return True

        for line in file:
            if "REMARK 410" in line or "IMGT " in line:
                file.close()
                return True


            elif "ATOM      1" in line:
                file.close()
                return False

        file.close()
        return False


    def generate_PDB_Structures(self):
        list_of_PDBStructures = []

        for pdb_filepath in self.get_pdb_filepaths():
            pdb_Parser = PDBParser(
                PERMISSIVE=True
            )

            structure = pdb_Parser.get_structure("pdbStructure", pdb_filepath)
            if self.check_if_PDB_file_has_IMGT_numbering(pdb_filepath):
                list_of_PDBStructures.append(PDBStructure(structure, contains_antigen=False, receptor_type="BCR",
                                                          has_imgt_numbering=True, region_type=self.region_type))

            else:
                id_of_file = self.pdb_file_paths.index(pdb_filepath)

                try:
                    start_position_from_meta_file = self.get_start_and_stop_position_from_metafile(id_of_file)
                    stop_position_from_meta_file = self.get_start_and_stop_position_from_metafile(id_of_file)
                    list_of_PDBStructures.append(PDBStructure(structure, contains_antigen=False, receptor_type="BCR",
                                                              has_imgt_numbering=False, start_position=start_position_from_meta_file[0],
                                                              stop_position=stop_position_from_meta_file[1], region_type=self.region_type))
                except:
                    list_of_PDBStructures.append(PDBStructure(structure, contains_antigen=False, receptor_type="BCR",
                                                              has_imgt_numbering=False,
                                                              region_type=self.region_type))
                    print("Start and stop position column are required in the metadata file for non imgt-numbered PDB files - ", pdb_filepath)

        return list_of_PDBStructures


    def get_start_and_stop_position_from_metafile(self,id_of_file):
        start_position = self.get_metadata(["start_position"]).get("start_position")[id_of_file]
        stop_position = self.get_metadata(["stop_position"]).get("stop_position")[id_of_file]
        return start_position, stop_position


    def get_id_by_pdb_structure(self, name_of_pdb_structure):
        for id_of_file in range(0, len(self.pdb_file_paths)):
            if name_of_pdb_structure.lower() in self.pdb_file_paths[id_of_file]:
                return id_of_file



    def get_PDB_Parser(self):
        for current_file in self.get_files():
            pdb_parser = PDBParser(
                PERMISSIVE=True
            )
            pdb_structure = pdb_parser.get_structure("pdbStructure", current_file)
            yield pdb_structure

    def get_data(self):
        for files in self.list_of_PDB_Structures:
            yield files


    def get_list_of_PDB_structures(self):
        return self.list_of_PDB_Structures


    def get_files(self):
        for files in self.pdb_file_paths:
            yield files


    def get_filenames(self):
        return self.file_names

    def set_filenames(self, filenames):
        self.file_names = filenames

    def set_pdb_filepaths(self, filepaths):
        self.pdb_file_paths = filepaths

    def get_example_count(self):
        return len(self.pdb_file_paths)

    def get_example_ids(self):
        example_ids =[]
        for files in self.file_names:
            file_name = Path(files).name.split(".")[0]
            example_ids.append(file_name)

        return example_ids


    def get_label_names(self):
        """Returns the list of metadata fields which can be used as labels"""
        return self.labels

    def clone(self, keep_identifier: bool = False):
        raise NotImplementedError

    def get_metadata(self, field_names: list, return_df: bool = False):
        assert isinstance(self.metadata_file, Path) and self.metadata_file.is_file(), \
            f"PDBDataset: for dataset {self.name} (id: {self.identifier}) metadata file is not set properly. The metadata file points to " \
            f"{self.metadata_file}."

        df = pd.read_csv(self.metadata_file, sep=",", usecols=field_names, comment=Constants.COMMENT_SIGN)
        if return_df:
            return df
        else:
            return df.to_dict("list")

    def get_pdb_filepaths(self):
        return self.pdb_file_paths

    def get_metadata_file(self):
        return self.metadata_file


    def make_subset(self, example_indices, path: Path, dataset_type: str):

        file_names = []
        for index in example_indices:
            file_names.append(self.file_names[index])


        metadata_file = self._build_new_metadata(example_indices, path / f"{dataset_type}_metadata.csv")
        new_dataset = PDBDataset(pdb_file_paths=[self.pdb_file_paths[i] for i in example_indices],file_names= file_names, labels=copy.deepcopy(self.labels),
                                 metadata_file=metadata_file)

        return new_dataset

    def _build_new_metadata(self, indices, path: Path) -> Path:
        if self.metadata_file:
            df = pd.read_csv(self.metadata_file, comment=Constants.COMMENT_SIGN)
            df = df.iloc[indices, :]
            df.to_csv(path, index=False)
            return path
        else:
            return None