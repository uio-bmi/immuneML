import numpy
from Bio.PDB import PDBParser

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
import numpy as np
from immuneML.data_model.receptor.RegionType import RegionType
import tmscoring
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select





class DistanceBetweenStructuresEncoder(DatasetEncoder):



    @staticmethod
    def build_object(dataset=None, **params):
        return DistanceBetweenStructuresEncoder(**params)

    def __init__(self, name: str = None, region_type: RegionType = None, context: dict = None):

        self.name = name
        self.region_type = region_type
        self.context = context

    def set_context(self, context: dict):
        self.context = context
        return self


    def encode(self, dataset, params: EncoderParams):

        print("")

        dataset.set_filenames(self.create_new_pdb_file_without_antigen(dataset))

        distance_matrix = self.build_distance_matrix(dataset,params)

        encoded_dataset = PDBDataset(dataset.pdb_file_paths, dataset.file_names, dataset.labels, dataset.metadata_file, EncodedData(distance_matrix, None, dataset.get_example_ids()))

        return encoded_dataset


    def create_new_pdb_file_without_antigen(self, dataset):

        new_file_paths = []

        for pdb_file_path in dataset.pdb_file_paths:
            parser = PDBParser()
            structure = parser.get_structure("pdbStructure", pdb_file_path)

            io = PDBIO()
            io.set_structure(structure)

            new_file_name = pdb_file_path + "_removed_antigen"
            io.save(new_file_name, chain_Select())
            new_file_paths.append(new_file_name)

        return new_file_paths





    def build_distance_matrix(self, dataset: PDBDataset, params: EncoderParams):

        entire_dataset = dataset if self.context is None or "dataset" not in self.context else self.context["dataset"]

        print("hey")
        distance_matrix = self.calculate_TM_Score_between_current_dataset_and_entire_dataset(dataset, entire_dataset)

        return distance_matrix

    def calculate_TM_Score_between_current_dataset_and_entire_dataset(self, current_PDB_structures, entire_dataset):

        entire_dataset.set_filenames(self.create_new_pdb_file_without_antigen(entire_dataset))
        distance_matrix = []

        for current_pdb_file in current_PDB_structures.get_filenames():
            current_structure_matrix = []
            for compare_to_pdb_file in entire_dataset.get_filenames():
                alignment = tmscoring.TMscoring(current_pdb_file, compare_to_pdb_file)
                current_structure_matrix.append(alignment.tmscore(**alignment.get_current_values()))

            distance_matrix.append(current_structure_matrix)

        return distance_matrix


class chain_Select(Select):
    def accept_chain(self, chain):
        if chain._id == 'L' or chain._id == 'H':
            return True
        else:
            return False

