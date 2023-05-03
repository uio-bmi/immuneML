import pandas as pd
import pathlib

from Bio.PDB import PDBParser

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.data_model.pdb_structure.PDBStructure import PDBStructure
from immuneML.data_model.receptor.RegionType import RegionType
from scripts.specification_util import update_docs_per_mapping
from immuneML.data_model.receptor.RegionType import RegionType


class PDBImport(DataImport):


    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:

        current_path = params.get("path")


        paths_to_pdb_structures = []
        file_names = []


        for pdb_file in pathlib.Path(str(current_path)).glob('*.pdb'):
            paths_to_pdb_structures.append(str(pdb_file))
            file_names.append(str(pdb_file.name))

        dataframe = pd.read_csv(params.get("metadata_file"))
        labels = list(dataframe.columns)
     ##   labels = dict(zip(dataframe.filename, dataframe.sars))


        return PDBDataset(pdb_file_paths=paths_to_pdb_structures, file_names=file_names,labels={key: list(set(dataframe[key].values.tolist())) for key in labels}, metadata_file=params.get("metadata_file"), region_type=params.get("region_type"))


    @staticmethod
    def get_documentation():
        doc = str(PDBImport.__doc__)

        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
