from pathlib import Path
from uuid import uuid4
import pandas as pd
from immuneML.environment.Constants import Constants

import Bio

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from typing import List



class PDBDataset(Dataset):
    """
    A dataset for PDB files. The PDBDataset class is similar to the other dataset classes, it just handles data related
    to the given PDB files.
    """



    def __init__(self,pdbFilePaths: List= None, filenames: list = None, labels: dict = None, metadata_file: Path = None,encoded_data: EncodedData = None, identifier: str = None,
             name: str = None, element_ids: list = None ):
        super().__init__()
        self.encoded_data = encoded_data
        self.labels = labels
        self.identifier = identifier if identifier is not None else uuid4().hex
        self.name = name
        self.pdbFilePaths = pdbFilePaths
        self.filenames = filenames
        self.metadata_file = metadata_file




#Return an iterator of the pdb objects
    def get_data(self):
        for files in self.pdbFilePaths:
            yield files

    def get_filenames(self):
        return self.filenames

    def set_filenames(self, filenames):
        self.filenames = filenames

    def get_example_count(self):
        return len(self.get_example_ids())


    def get_example_ids(self):
        example_ids =[]
        for files in self.filenames:
            fileName = Path(files).name.split(".")[0]
            example_ids.append(fileName)

        return example_ids


    def get_label_names(self):
        """Returns the list of metadata fields which can be used as labels"""
        return self.labels

    def clone(self, keep_identifier: bool = False):
        raise NotImplementedError

    def get_metadata(self, field_names: list, return_df: bool = False):
        """
        A function to get the metadata of the repertoires. It can be useful in encodings or reports when the repertoire information needed is not
        present only in the label chosen for the ML model (e.g., disease), but also other information (e.g., age, HLA).

        Args:
            field_names (list): list of the metadata fields to return; the fields must be present in the metadata files. To find fields available, use :py:obj:`~immuneML.data_model.dataset.RepertoireDataset.RepertoireDataset.get_label_names` function.
            return_df (bool): determines if the results should be returned as a dataframe where each column corresponds to a field or as a dictionary

        Returns:

            a dictionary where keys are fields names and values are lists of field values for each repertoire; alternatively returns the same information in dataframe format

        """
        assert isinstance(self.metadata_file, Path) and self.metadata_file.is_file(), \
            f"PDBDataset: for dataset {self.name} (id: {self.identifier}) metadata file is not set properly. The metadata file points to " \
            f"{self.metadata_file}."

        df = pd.read_csv(self.metadata_file, sep=",", usecols=field_names, comment=Constants.COMMENT_SIGN)
        if return_df:
            return df
        else:
            return df.to_dict("list")

    def getPdbFilepaths(self):
        return self.pdbFilePaths

    def getFilenames(self):
        return self.filenames

    def getMetadataFile(self):
        return self.metadata_file