import pandas as pd

from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset import Dataset
from source.util.ImportHelper import ImportHelper


class OLGAImport(DataImport):
    """
    Imports data generated by OLGA simulations into a RepertoireDataset. Assumes one file per repertoire
    and that the columns in each file correspond to: nucleotide sequences, amino acid sequences, v genes, j genes

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_olga_dataset:
            format: OLGA
            params:
                # required parameters:
                metadata_file: path/to/metadata.csv
                path: path/to/repertoire/files/
                result_path: path/where/to/store/immuneML/imported/data/
                # these parameters need to be specified only if different than default behaviour is desired
                separator: "\\t"
                columns_to_load: null # OLGA columns are assumed to correspond to: sequences, sequence_aas, v_genes, j_genes
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(OLGAImport, params, dataset_name)


    @staticmethod
    def alternative_load_func(filepath, params):
        df = pd.read_csv(filepath, sep=params.separator, iterator=False, dtype=str, header=None)
        df.columns = ["sequences", "sequence_aas", "v_genes", "j_genes"]
        return df


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        if "sequences" not in df.columns and "sequence_aas" not in df.columns:
            raise IOError("OLGAImport: Columns should contain at least 'sequences' or 'sequence_aas'.")

        if "counts" not in df.columns:
            df["counts"] = 1

        df["sequence_identifiers"] = None

        ImportHelper.junction_to_cdr3(df, params.region_type)

        return df

