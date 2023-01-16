import numpy
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
import numpy as np
from immuneML.data_model.receptor.RegionType import RegionType



class PDBDistanceMatrixEncoder(DatasetEncoder):



    @staticmethod
    def build_object(dataset=None, **params):
        return PDBDistanceMatrixEncoder(**params)

    def __init__(self, name: str = None, region_type: RegionType = None):

        self.name = name
        self.region_type = region_type

    def encode(self, dataset, params: EncoderParams):

        processed_data = self.extract_data_for_encoding(dataset, params)

        encoded_dataset = PDBDataset(dataset.pdb_file_paths, dataset.file_names, dataset.labels, dataset.metadata_file, EncodedData(processed_data, None, dataset.get_example_ids()))

        return encoded_dataset


    def extract_data_for_encoding(self, dataset, params: EncoderParams):

        structures =[]

        for PDB_structure in dataset.list_of_PDB_Structures:
            collection = self.get_carbon_alphas(PDB_structure)

            chain_IDs = self.get_chain_ID(collection)

            light_id = chain_IDs[0]
            heavy_id = chain_IDs[1]
            antigen_id = chain_IDs[2]


            light_chain_distance_to_antigen = np.zeros(shape=(len(collection[antigen_id]), len(collection[light_id])))

            self.caluculate_atom_distance(collection, light_id, antigen_id, light_chain_distance_to_antigen)

            heavy_chain_distance_to_antigen = np.zeros(shape=(len(collection[antigen_id]), len(collection[heavy_id])))
            self.caluculate_atom_distance(collection, heavy_id, antigen_id, heavy_chain_distance_to_antigen)

            padded_arrays = self.pad_chains_to_same_length(light_chain_distance_to_antigen, heavy_chain_distance_to_antigen)

            structures.append([padded_arrays[0],padded_arrays[1]])


        padded_structures = self.find_longest_antigen_then_pad_to_that_length(structures)

        numpy_array = np.array(padded_structures)

        return numpy_array


    def get_carbon_alphas(self, PDB_structure):
        if PDB_structure.get_has_imgt_numbering():
            return self.get_carbon_alphas_with_IMGT_numbering(PDB_structure.get_pdb_structure())
        else:
            return self.get_carbon_alphas_with_start_stop(PDB_structure)



    def get_carbon_alphas_with_start_stop(self, PDB_structure):
        collection = []
        list = []
        counter = 0

        for model in PDB_structure.get_pdb_structure():
            for chain in model:
                residue_counter = 0
                for residue in chain:
                    if residue_counter == 0:
                        list.append(residue.get_parent())
                        residue_counter = 1

                    for atom in residue:
                        if (atom.get_name() == "CA"):
                            if (self.is_in_start_and_stop_positions(atom.full_id[3][1], PDB_structure.get_start_position(), PDB_structure.get_stop_position())) or len(collection) >= 2:
                                list.append(atom.get_coord())

                save_list = list.copy()
                collection.append(save_list)
                counter = counter + 1
                list.clear()

        return collection

    def get_carbon_alphas_with_IMGT_numbering(self, parser_object):
        collection = []
        list = []

        region = self.region_type
        counter = 0

        for model in parser_object:
            for chain in model:
                residue_counter = 0
                for residue in chain:
                    if residue_counter == 0:
                        list.append(residue.get_parent())
                        residue_counter = 1

                    for atom in residue:
                        if (atom.get_name() == "CA"):

                            if "FULL_SEQUENCE" in region:
                                list.append(atom.get_coord())

                            else:
                                if "IMGT_CDR3" in region and (
                                        self.is_in_CDR3(atom.full_id[3][1]) or len(collection) >= 2):
                                    list.append(atom.get_coord())

                                if "IMGT_CDR2" in region and (
                                        self.is_in_CDR2(atom.full_id[3][1]) or len(collection) >= 2):
                                    list.append(atom.get_coord())

                                if "IMGT_CDR1" in region and (
                                        self.is_in_CDR1(atom.full_id[3][1]) or len(collection) >= 2):
                                    list.append(atom.get_coord())

                save_list = list.copy()
                collection.append(save_list)
                counter = counter + 1
                list.clear()

        return collection

    def caluculate_atom_distance(self, collection, chain, antigen, list):
        for x in range(len(collection[antigen])):
            for i in range(len(collection[chain])):
                a = np.array((collection[chain][i][0], collection[chain][i][1], collection[chain][i][2]))
                b = np.array((collection[antigen][x][0], collection[antigen][x][1], collection[antigen][x][2]))

                rounded = round(np.linalg.norm(a - b), 2)
                list[x][i] = rounded



    def find_longest_antigen_then_pad_to_that_length(self, structures):
        longest_antigen = 0
        longest_chain = 0

        for x in structures:
            if len(x[0]) > longest_antigen:
                longest_antigen = len(x[0])
            if len(x[0][0]) > longest_chain:
                longest_chain = len(x[0][0])

        padded_structures = []
        for x in structures:
            padded_structures.append(self.pad_antigens_to_same_length(x, longest_antigen, longest_chain))

        return  padded_structures


    def pad_chains_to_same_length(self, light_chain, heavy_chain):
        diff = len(light_chain[0]) - len(heavy_chain[0])

        if diff > 0:
            longest = light_chain
            shortest = heavy_chain
            padded_shortest = []
            for x in shortest:
                padded_array = np.pad(x, (0, diff), 'constant', constant_values=(0, np.inf))
                padded_shortest.append(padded_array)

            padded_shortest_to_numpy_array = np.array(padded_shortest)
            return (light_chain, padded_shortest_to_numpy_array)

        elif diff < 0:
            longest= heavy_chain
            shortest= light_chain
            diff = abs(diff)

            padded_shortest = []
            for x in shortest:
                padded_array = np.pad(x, (0, diff), 'constant', constant_values=(0, np.inf))
                padded_shortest.append(padded_array)

            padded_shortest_to_numpy_array = np.array(padded_shortest)
            return (padded_shortest_to_numpy_array, heavy_chain)

        else:
            return (light_chain, heavy_chain)


    def pad_antigens_to_same_length(self, chains, antigen_length, chain_length):
        return_array = chains.copy()
        if chain_length > len(chains[0][0]):
            for x in range(0, len(chains)):
                padded_chain = []
                chain_diff = chain_length - len(chains[x][0])

                if chain_diff > 0:
                    padded_array_columns = []
                    for y in chains[x]:
                        padded_array = np.pad(y, (0, chain_diff), 'constant', constant_values=(0, np.inf))
                        padded_array_columns.append(padded_array)

                    padded_chain = np.array(padded_array_columns)
                return_array[x] = padded_chain

        if antigen_length > len(chains[0]):
            for i in range(0, len(chains)):
                antigen_diff = antigen_length - len(chains[i])

                if antigen_diff > 0:
                    for j in range(0, antigen_diff):
                        empty_arr = np.full((1, chain_length), np.inf)
                        return_array[i] = numpy.append(return_array[i], empty_arr, 0)

        return return_array





    def get_chain_ID(self, collection):
        light_id = -1
        heavy_id = -1
        antigen_id = -1

        for i in range(0, len(collection)):
            chain_id = collection[i][0].id

            if 'L' in chain_id or 'A' in chain_id:
                light_id = i

            elif 'H' in chain_id or 'B' in chain_id:
                heavy_id = i

            elif 'P' in chain_id or 'C' in chain_id:
                antigen_id = i

            collection[i].pop(0)

        if light_id >= 0 and heavy_id >= 0 and antigen_id == -1:
            if light_id + heavy_id == 1:
                antigen_id = 2

            elif light_id + heavy_id == 3:
                antigen_id = 0

        elif light_id == -1 or heavy_id == -1 or antigen_id == -1:
            light_id = 0
            heavy_id = 1
            antigen_id = 2

        return light_id, heavy_id, antigen_id


    def is_in_CDR3(self, num):
        if 117 >= num >= 105:
            return True
        else:
            return False


    def is_in_CDR2(self, num):
        if 65 >= num >= 56:
            return True
        else:
            return False


    def is_in_CDR1(self, num):
        if 38 >= num >= 27:
            return True
        else:
            return False

    def is_in_start_and_stop_positions(self, num, start, stop):
        if stop >= num >= start:
            return True
        else:
            return False
