import pandas as pd
from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset import Dataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class AdaptiveBiotechImport(DataImport):
    """
    Specification:

    .. indent with spaces
    .. code-block:: yaml

        my_adaptive_dataset:
            format: AdaptiveBiotech
            params:
                # these parameters have to be always specified:
                metadata_file: path/to/metadata.csv # csv file with fields filename, subject_id and arbitrary others which can be used as labels in analysis
                path: path/to/location/of/repertoire/files/ # all repertoire files need to be in the same folder to be loaded (they will be discovered based on the metadata file)
                result_path: path/where/to/store/imported/repertoires/ # immuneML imports data to optimized representation to speed up analysis so this defines where to store these new representation files
                # the following are default values so these need to be specified only if a different behavior is required
                import_productive: True
                import_with_stop_codon: False
                import_out_of_frame: False
                separator: "\\t"
                columns_to_load: [rearrangement, v_family, v_gene, v_allele, j_family, j_gene, j_allele, amino_acid, templates, frame_type, locus]
                column_mapping: # adaptive column names -> immuneML repertoire fields
                    rearrangement: sequences # 'rearrangement' is the adaptive name, which will be mapped to 'sequences' in immuneML
                    amino_acid: sequence_aas
                    v_gene: v_genes
                    j_gene: j_genes
                    frame_type: frame_types
                    v_family: v_subgroup
                    j_family: j_subgroup
                    templates: counts
                    locus: chains

    """


    @staticmethod
    def import_dataset(params: dict, dataset_name: str) -> Dataset:
        return ImportHelper.import_dataset(AdaptiveBiotechImport, params, dataset_name)


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, params: DatasetImportParams):
        df["frame_types"] = df.frame_types.str.upper()

        frame_type_list = ImportHelper.prepare_frame_type_list(params)
        df = df[df["frame_types"].isin(frame_type_list)]

        if params.region_type == RegionType.IMGT_CDR3:
            if "sequences" in params.columns_to_load:
                df['sequences'] = [y[(84 - 3 * len(x)): 78] for x, y in zip(df['sequence_aas'], df['sequences'])]
            df['sequence_aas'] = df["sequence_aas"].str[1:-1]
            df["region_types"] = RegionType.IMGT_CDR3.name
        elif "sequences" in params.columns_to_load:
            df['sequences'] = [y[(81 - 3 * len(x)): 81] for x, y in zip(df['sequence_aas'], df['sequences'])] # todo check this?

        df = ImportHelper.parse_adaptive_germline_to_imgt(df)
        df = ImportHelper.standardize_none_values(df)

        return df

    @staticmethod
    def import_receptors(df, params):
        raise NotImplementedError("AIRRImport: import of paired receptor AIRR data has not been implemented.")


