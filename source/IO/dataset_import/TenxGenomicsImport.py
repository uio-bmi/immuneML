from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.util.ImportHelper import ImportHelper
import pandas as pd

class TenxGenomicsImport(DataImport):
    """
    Imports data from 10xGenomics. The files that should be used as input are named 'Clonotype consensus annotations (CSV)'.

    Note: by default the 10xGenomics field umis is used to define the immuneML field counts. If you want to use the 10xGenomics
    field reads instead, this can be changed in the column_mapping (set reads: counts).
    Furthermore, the 10xGenomics field clonotype_id is used for the immuneML field cell_id.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_10xGenomics_dataset:
            format: TenxGenomics
            params:
                # required parameters:
                metadata_file: path/to/metadata.csv
                path: path/to/directory/with/repertoire/files/
                result_path: path/where/to/store/imported/repertoires/
                # optional parameters (if not specified the values bellow will be used):
                import_productive: True # whether to import productive sequences
                import_unproductive: False # whether to import unproductive sequences
                region_type: "IMGT_CDR3" # which part of the sequence to import by default
                columns_to_load: [clonotype_id, consensus_id, length, chain, v_gene, d_gene, j_gene, c_gene, full_length, productive, cdr3, cdr3_nt, reads, umis]
                    cdr3: sequence_aas
                    cdr3_nt: sequences
                    v_gene: v_genes
                    j_gene: j_genes
                    umis: counts
                    chain: chains
                    clonotype_id: cell_ids
                    consensus_id: sequence_identifiers
                separator: ","
    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(TenxGenomicsImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        df["frame_types"] = None
        df.loc[df["productive"].eq("True"), "frame_types"] = SequenceFrameType.IN.name

        allowed_productive_values = []
        if params.import_productive:
            allowed_productive_values.append("True")
        if params.import_unproductive:
            allowed_productive_values.append("False")

        df = df[df.productive.isin(allowed_productive_values)]

        ImportHelper.junction_to_cdr3(df, params.region_type)
        return df


    @staticmethod
    def import_receptors(df, params):
        df["receptor_identifiers"] = df["cell_ids"]
        return ImportHelper.import_receptors(df, params)


