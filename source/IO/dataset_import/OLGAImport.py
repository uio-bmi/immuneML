import pandas as pd

from scripts.specification_util import update_docs_per_mapping
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset import Dataset
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class OLGAImport(DataImport):
    """
    Imports data generated by OLGA simulations into a Repertoire-, or SequenceDataset. Assumes that the columns in each
    file correspond to: nucleotide sequences, amino acid sequences, v genes, j genes


    Arguments:

        path (str): This is the path to a directory with OLGA files to import. By default path is set to the current working directory.

        is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset.
        By default, is_repertoire is set to True.

        metadata_file (str): Required for RepertoireDatasets. This parameter specifies the path to the metadata file.
        This is a csv file with columns filename, subject_id and arbitrary other columns which can be used as labels in instructions.
        Only the OLGA files included under the column 'filename' are imported into the RepertoireDataset.
        SequenceDataset metadata is currently not supported.

        region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the
        first and last amino acids are removed from the CDR3 sequence, as OLGA uses the IMGT junction. Specifying
        any other value will result in importing the sequences as they are.
        Valid values for region_type are the names of the :py:obj:`~source.data_model.receptor.RegionType.RegionType` enum.

        separator (str): Column separator, for OLGA this is by default "\\t".


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_olga_dataset:
            format: OLGA
            params:
                path: path/to/files/
                is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
                metadata_file: path/to/metadata.csv # metadata file for RepertoireDataset
                # Optional fields with OLGA-specific defaults, only change when different behavior is required:
                separator: "\\t" # column separator
                region_type: IMGT_CDR3 # what part of the sequence to import
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
        ImportHelper.drop_empty_sequences(df)

        return df


    @staticmethod
    def get_documentation():
        doc = str(OLGAImport.__doc__)

        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for region_type are the names of the :py:obj:`~source.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc

