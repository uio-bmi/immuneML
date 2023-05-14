import os
import unittest

from Bio.PDB import PDBParser

from immuneML.data_model.dataset.PDBDataset import PDBDataset


class TestPDBDataset(unittest.TestCase):

    pdb_parser = None
    path = None
    pdb_dataset = None
    pdb_structure = None


    @classmethod
    def setUpClass(cls):
        cls.pdb_parser = PDBParser(
            PERMISSIVE=True
        )
        cls.pdb_structure = cls.pdb_parser.get_structure("pdbStructure", "1u8o.pdb")
        cls.path = ["1u8o.pdb"]
        cls.pdb_dataset = PDBDataset(pdb_file_paths=cls.path,
                                 file_names="1u8o.pdb")


    def test_generate_PDB_Structures(self):
        self.assertEqual(self.pdb_dataset.get_list_of_PDB_structures()[0].pdb_structure, self.pdb_structure)



    def test_check_if_PDB_file_has_IMGT_numbering(self):
        self.assertTrue(self.pdb_dataset.check_if_PDB_file_has_IMGT_numbering(self.path[0]))

