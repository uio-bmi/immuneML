from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.util.ImportHelper import ImportHelper


class ImmunoSEQImport(DataImport):
    """
    Imports data from immunoSEQ format from Adaptive. Very similar to AdaptiveBiotechImport, except that column names in the original files
    are different.

    Specification:
        path: ./data/ # path where to find the repertoire files in the given format
        result_path: ./result/ # where to store the imported files
        metadata_file: ./data/metadata.csv # path to metadata file, for more information on the format, see the documentation
        region_definition: "IMGT" # which CDR3 definition to use - IMGT option means removing first and last amino acid as ImmunoSEQ uses IMGT junction as CDR3
        separator: '\t'
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
    def import_dataset(params: dict) -> Dataset:
        immunoseq_params = DatasetImportParams.build_object(**params)
        return ImportHelper.import_repertoire_dataset(ImmunoSEQImport.preprocess_repertoire, immunoseq_params)

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams):

        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)

        if params.region_definition == RegionDefinition.IMGT:
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
        else:
            df['sequences'] = [y[(84 - 3 * len(x)): 78] if x is not None else None for x, y in zip(df['sequence_aas'], df['sequences'])]

        return df
