from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper
import pandas as pd

class ImmunoSEQImport(DataImport):
    """
    Imports data from immunoSEQ format from Adaptive. Very similar to AdaptiveBiotech format, except that column names in the original files
    are different.

    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_immunoseq_dataset:
            format: ImmunoSEQ
            params:
                path: ./data/ # path where to find the repertoire files in the given format
                result_path: ./result/ # where to store the imported files
                metadata_file: ./data/metadata.csv # path to metadata file, for more information on the format, see the documentation
                region_definition: "IMGT" # which CDR3 definition to use - IMGT option means removing first and last amino acid as ImmunoSEQ uses IMGT junction as CDR3
                separator: '\\t'
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

        df["frame_types"] = df.frame_types.str.upper()

        frame_type_list = ImportHelper.prepare_frame_type_list(params)
        df = df[df["frame_types"].isin(frame_type_list)]

        if params.region_definition == RegionDefinition.IMGT and params.region_type == RegionType.CDR3:
            if "sequences" in df.columns and "sequence_aas" in df.columns:
                df = ImmunoSEQImport.extract_sequence_from_rearrangement(df, params.region_definition)
            elif "sequence_aas" in df.columns:
                df['sequence_aas'] = df["sequence_aas"].str[1:-1]
        elif "sequences" in df.columns and "sequence_aas" in df.columns:
            df = ImmunoSEQImport.extract_sequence_from_rearrangement(df, params.region_definition)

        df = ImportHelper.parse_adaptive_germline_to_imgt(df)

        return df

    @staticmethod
    def extract_sequence_from_rearrangement(df, region_definition: RegionDefinition):

        if region_definition == RegionDefinition.IMGT:
            df['sequences'] = [y[(81 - 3 * len(x)): 81] if x is not None else None for x, y in zip(df['sequence_aas'], df['sequences'])]
            df['sequence_aas'] = df["sequence_aas"].str[1:-1]
        else:
            df['sequences'] = [y[(84 - 3 * len(x)): 78] if x is not None else None for x, y in zip(df['sequence_aas'], df['sequences'])]

        return df


    @staticmethod
    def import_receptors(df, params):
        raise NotImplementedError("ImmunoSEQImport: import of paired receptor ImmunoSEQ data has not been implemented.")





