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

    Specification:

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
                import_productive: True # whether to only import productive sequences
                region_type: "CDR3" # which part of the sequence to import by default
                region_definition: "IMGT" # which CDR3 definition to use - IMGT option means removing first and last amino acid as 10xGenomics uses IMGT junction as CDR3
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
        tenx_params = DatasetImportParams.build_object(**params)

        dataset = ImportHelper.load_dataset_if_exists(params, tenx_params, dataset_name)

        if dataset is None:
            if tenx_params.is_repertoire:
                dataset = ImportHelper.import_repertoire_dataset(TenxGenomicsImport.preprocess_repertoire, tenx_params, dataset_name)
            else:
                dataset = ImportHelper.import_sequence_dataset(TenxGenomicsImport.import_items, tenx_params, dataset_name)
        return dataset


    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams):
        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)
        df = TenxGenomicsImport.preprocess_dataframe(df, params)
        return df

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        df["frame_types"] = SequenceFrameType.IN.name # todo we only know productive/unproductive, not specific frame type

        if params.import_productive:
            df = df[df.productive.eq("True")]
        else:
            df.loc[df["productive"].eq("False"), "frame_types"] = SequenceFrameType.OUT.name

        ImportHelper.junction_to_cdr3(df, params.region_definition, params.region_type)
        return df


    @staticmethod
    def import_items(path, params):
        df = ImportHelper.load_sequence_dataframe(path, params)
        df = TenxGenomicsImport.preprocess_dataframe(df, params)

        if params.paired:
            df["receptor_identifiers"] = df["cell_ids"]
            sequences = ImportHelper.import_receptors(df, params)
        else:
            sequences = df.apply(ImportHelper.import_sequence, axis=1).values

        return sequences
