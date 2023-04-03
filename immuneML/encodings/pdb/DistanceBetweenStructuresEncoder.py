from Bio.PDB import PDBParser
from immuneML.encodings.distance_encoding.DistanceMetricType import DistanceMetricType

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
import numpy as np
from immuneML.data_model.receptor.RegionType import RegionType
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select

from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
import pandas as pd
from immuneML.util.EncoderHelper import EncoderHelper
from pathlib import Path
from immuneML.caching.CacheHandler import CacheHandler


class DistanceBetweenStructuresEncoder(DatasetEncoder):

    @staticmethod
    def build_object(dataset=None, **params):
        return DistanceBetweenStructuresEncoder(**params)

    def __init__(self, distance_metric: DistanceMetricType, name: str = None, region_type: RegionType = None, context: dict = None):

        self.name = name
        self.region_type = region_type
        self.context = context
        self.distance_fn = distance_metric


    def set_context(self, context: dict):
        self.context = context
        return self


    def encode(self, dataset, params: EncoderParams):

        training_data_ids = EncoderHelper.prepare_training_ids(dataset, params)

        dataset.set_pdb_filepaths(self.create_new_pdb_file_without_antigen(dataset))

        distance_matrix = self.build_distance_matrix(dataset, training_data_ids, params)

        labels = self.build_labels(dataset, params) if params.encode_labels else None

        encoded_dataset = PDBDataset(dataset.pdb_file_paths, dataset.file_names, dataset.labels, dataset.metadata_file, EncodedData(examples=distance_matrix, labels=labels, example_ids=dataset.get_example_ids(), encoding=DistanceBetweenStructuresEncoder.__name__))

        return encoded_dataset


    def create_new_pdb_file_without_antigen(self, dataset):

        new_file_paths = []

        for pdb_file_path in dataset.pdb_file_paths:

            if "_removed_antigen.pdb" in pdb_file_path:
                new_file_paths.append(pdb_file_path)

            else:
                file_name = Path(pdb_file_path).name.split(".")[0]

                parser = PDBParser()
                structure = parser.get_structure("pdbStructure", pdb_file_path)

                io = PDBIO()
                io.set_structure(structure)

                new_file_name = file_name + "_removed_antigen.pdb"
                io.save(new_file_name, chain_Select())
                new_file_paths.append(new_file_name)

        return new_file_paths





    def build_distance_matrix(self, dataset: PDBDataset, training_data_ids, params):

        entire_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        example_ids_with_correct_endings = self.add_file_endings(dataset.get_example_ids())
        training_data_ids_with_correct_endings = self.add_file_endings(training_data_ids)

        distance_matrix_dataframe = CacheHandler.memo_by_params((("dataset_identifier", dataset.identifier),
                                            "pdb_dataset_comparison",
                                            ("comparison_fn", self.distance_fn)),
                                           lambda: self.calculate_TM_Score_between_current_dataset_and_entire_dataset(entire_dataset, entire_dataset))

        distance_matrix_dataframe = distance_matrix_dataframe.loc[example_ids_with_correct_endings, training_data_ids_with_correct_endings]

        return distance_matrix_dataframe



    def calculate_TM_Score_between_current_dataset_and_entire_dataset(self, current_PDB_structures, entire_dataset):

        entire_dataset.set_pdb_filepaths(self.create_new_pdb_file_without_antigen(entire_dataset))
        distance_dictionary = {}


        for path_to_current_pdb_file in current_PDB_structures.get_pdb_filepaths():
            current_structure_matrix = []
            for path_to_pdb_file_to_compare in current_PDB_structures.get_pdb_filepaths():

                current_structure_matrix.append(self.calcualte_TM_Score_between_two_pdb_files(path_to_current_pdb_file,path_to_pdb_file_to_compare))

            distance_dictionary[path_to_current_pdb_file] = current_structure_matrix

        distance_matrix_dataframe = pd.DataFrame(distance_dictionary)
        distance_matrix_dataframe.index = entire_dataset.get_pdb_filepaths()

        return distance_matrix_dataframe


    def calcualte_TM_Score_between_two_pdb_files(self, path_to_current_pdb_file, path_to_other_pdb_file):
        current_pdb_structure = get_structure(path_to_current_pdb_file)
        other_pdb_structure = get_structure(path_to_other_pdb_file)

        chains_of_current_pdb_structure = next(current_pdb_structure.get_chains())
        chains_of_other_pdb_structure = next(other_pdb_structure.get_chains())

        coords_from_current_pdb_structure, seq_of_current_pdb_structure = get_residue_data(chains_of_current_pdb_structure)
        coords_from_other_pdb_structure, seq_of_other_pdb_structure = get_residue_data(chains_of_other_pdb_structure)

        tm_score = tm_align(coords_from_current_pdb_structure, coords_from_other_pdb_structure, seq_of_current_pdb_structure, seq_of_other_pdb_structure)

        return max(tm_score.tm_norm_chain1,tm_score.tm_norm_chain2)

    def build_labels(self, dataset: PDBDataset, params: EncoderParams) -> dict:

        tmp_labels = dataset.get_metadata(params.label_config.get_labels_by_name(), return_df=True)
        tmp_labels = tmp_labels.to_dict("list")

        return tmp_labels

    def add_file_endings(self, list_of_files):

        files_with_added_endings = []
        for file in list_of_files:
            if ".pdb" in file:
                new_file_name = Path(file).name.split(".")[0]
                files_with_added_endings.append(new_file_name[0] + "_removed_antigen.pdb")

            else:
                files_with_added_endings.append(file + "_removed_antigen.pdb")

        return files_with_added_endings

class chain_Select(Select):
    def accept_chain(self, chain):
        if chain._id == 'L' or chain._id == 'H':
            return True
        else:
            return False

