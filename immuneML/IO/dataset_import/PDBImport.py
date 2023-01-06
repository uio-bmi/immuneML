import airr
import pandas as pd
import pathlib


from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from scripts.specification_util import update_docs_per_mapping




class PDBImport(DataImport):


    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:

        curPath = params.get("path")


        pathsOfPDBStructures = []
        fileNames = []


        for pdb_file in pathlib.Path(str(curPath)).glob('*.pdb'):
            pathsOfPDBStructures.append(str(pdb_file))
            fileNames.append(str(pdb_file))

        df = pd.read_csv(params.get("metadata_file"))
        labels = list(df.columns)

        return PDBDataset(pathsOfPDBStructures, fileNames,labels, params.get("metadata_file"))

    @staticmethod
    def get_documentation():
        doc = str(PDBImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")
        repertoire_fields = list(Repertoire.FIELDS)
        repertoire_fields.remove("region_types")

        mapping = {
            "Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
            "Valid immuneML fields that can be specified here are defined by Repertoire.FIELDS": f"Valid immuneML fields that can be specified here are {repertoire_fields}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
