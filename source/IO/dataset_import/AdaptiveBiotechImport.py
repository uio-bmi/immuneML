from source.IO.dataset_import.DataImport import DataImport
from source.IO.dataset_import.DatasetImportParams import DatasetImportParams
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType
from source.util.ImportHelper import ImportHelper


class AdaptiveBiotechImport(DataImport):
    """
    YAML specification:

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
                region_definition: "IMGT" # which CDR3 definition to use - IMGT option means removing first and last amino acid as Adaptive uses IMGT junction as CDR3
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
    def import_dataset(params: dict, dataset_name: str) -> RepertoireDataset:
        adaptive_params = DatasetImportParams.build_object(**params)
        dataset = ImportHelper.import_or_load_imported(params, adaptive_params, dataset_name, AdaptiveBiotechImport.preprocess_repertoire)
        return dataset

    @staticmethod
    def preprocess_repertoire(metadata: dict, params: DatasetImportParams) -> dict:

        df = ImportHelper.load_repertoire_as_dataframe(metadata, params)

        frame_type_list = ImportHelper.prepare_frame_type_list(params)
        df = df[df["frame_types"].isin(frame_type_list)]

        if params.region_definition == RegionDefinition.IMGT:
            if "sequences" in params.columns_to_load:
                df['sequences'] = [y[(84 - 3 * len(x)): 78] for x, y in zip(df['sequence_aas'], df['sequences'])]
            df['sequence_aas'] = df["sequence_aas"].str[1:-1]
        elif "sequences" in params.columns_to_load:
            df['sequences'] = [y[(81 - 3 * len(x)): 81] for x, y in zip(df['sequence_aas'], df['sequences'])]

        df = ImportHelper.parse_adaptive_germline_to_imgt(df)

        df["region_types"] = RegionType.CDR3.name

        return df
