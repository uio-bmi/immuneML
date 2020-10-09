from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.util.AdaptiveImportHelper import AdaptiveImportHelper
from source.util.ImportHelper import ImportHelper
import pandas as pd

class ImmunoSEQImport(DataImport):
    """
    Imports data from immunoSEQ format from Adaptive. Very similar to AdaptiveBiotech format, except that column names in the original files
    are different.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_immunoseq_dataset:
            format: ImmunoSEQ
            params:
                path: ./data/ # path where to find the repertoire files in the given format
                result_path: ./result/ # where to store the imported files
                metadata_file: ./data/metadata.csv # path to metadata file, for more information on the format, see the documentation
                separator: '\\t'
                import_productive: True
                import_with_stop_codon: False
                import_out_of_frame: False
                columns_to_load: [nucleotide, aminoAcid, count (templates/reads), vFamilyName, vGeneName, vGeneAllele, jFamilyName, jGeneName, jGeneAllele, sequenceStatus] # columns from the original file that will be imported
                column_mapping: # immunoSEQ column names -> immuneML repertoire fields
                    nucleotide: sequences # 'nucleotide' is the immunoSEQ name, which will be mapped to 'sequences' in immuneML
                    aminoAcid: sequence_aas
                    vGeneName: v_genes
                    jGeneName: j_genes
                    sequenceStatus: frame_types
                    vFamilyName: v_subgroup
                    jFamilyName: j_subgroup
                    count (templates/reads): counts

    """

    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(ImmunoSEQImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        return AdaptiveImportHelper.preprocess_dataframe(df, params)