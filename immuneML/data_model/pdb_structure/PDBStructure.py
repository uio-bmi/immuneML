import Bio.PDB.Structure
from immuneML.data_model.DatasetItem import DatasetItem

from immuneML.data_model.receptor.RegionType import RegionType


class PDBStructure(DatasetItem):

    def get_attribute(self, name: str):
        pass

    def __init__(self, pdb_structure: Bio.PDB.Structure.Structure = None, contains_antigen: bool = False,
                 receptor_type: str = None,
                 region_type: RegionType = RegionType.IMGT_CDR3, ):
        self.pdb_structure = pdb_structure
        self.contains_antigen = contains_antigen
        self.receptor_type = receptor_type
        self.region_type = region_type

    def get_pdb_structure(self):
        return self.pdb_structure

    def get_contains_antigen(self):
        return self.contains_antigen

    def get_receptor_type(self):
        return self.receptor_type

    def get_region_type(self):
        return self.receptor_type
