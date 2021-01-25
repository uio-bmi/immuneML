import numpy as np
import pandas as pd
import yaml

from immuneML.data_model.encoded_data.EncodedData import EncodedData


def load_encoded_data(labels_path: str, encoding_details_path: str, design_matrix_path: str) -> EncodedData:
    """
    Utility function for adding ML methods; if one encodes data using immuneML through YAML specification and exports the
    encoded data using DesignMatrixExporter, this function can be used to import the data and return it in the format
    it would be available if the ML method was called from within immuneML

    Args:

        labels_path (str): path to labels file as exported by the DesignMatrixExporter
        encoding_details_path (str): path to the details file, where example_ids, feature_names and the encoding name will be imported from
        design_matrix_path (str): path to csv or npy file where the design matrix is stored

    Returns:

        EncodedData object as it would be provided to an ML method within immuneML

    """
    # read the data from these files
    examples = pd.read_csv(design_matrix_path).values if ".csv" in design_matrix_path[:-4] else np.load(design_matrix_path, allow_pickle=True)
    labels = pd.read_csv(labels_path).to_dict('list')

    with open(encoding_details_path, "r") as file:
        encoding_details = yaml.safe_load(file)

    # create an EncodedData object which can be used as an input argument for the fit or predict functions
    encoded_data = EncodedData(examples=examples, labels=labels,
                               example_ids=encoding_details['example_ids'],
                               feature_names=encoding_details['feature_names'],
                               encoding=encoding_details['encoding'])

    return encoded_data
