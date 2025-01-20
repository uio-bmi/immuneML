import json
import shutil
import zipfile
from pathlib import Path

import airr
import pandas as pd

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.data_model.SequenceParams import ChainPair, RegionType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.util.PathBuilder import PathBuilder
from scripts.specification_util import update_docs_per_mapping


class IReceptorImport(DataImport):
    """
    Imports AIRR datasets retrieved through the `iReceptor Gateway <https://gateway.ireceptor.org/home>`_ into a Repertoire-, Sequence- or ReceptorDataset.
    The differences between this importer and the :ref:`AIRR` importer are:

    * This importer takes in a list of .zip files, which must contain one or more AIRR tsv files, and for each AIRR file, a corresponding metadata json file must be present.
    * This importer does not require a metadata csv file for RepertoireDataset import, it is generated automatically from the metadata json files.

    RepertoireDatasets should be used when making predictions per repertoire, such as predicting a disease state.
    SequenceDatasets or ReceptorDatasets should be used when predicting values for unpaired (single-chain) and paired
    immune receptors respectively, like antigen specificity.

    AIRR rearrangement schema can be found here: https://docs.airr-community.org/en/stable/datarep/rearrangements.html

    When importing a ReceptorDataset, the AIRR field cell_id is used to determine the chain pairs.


    **Specification arguments:**

    - path (str): This is the path to a directory **with .zip files** retrieved from the iReceptor Gateway. These .zip files should include AIRR files (with .tsv extension) and corresponding metadata.json files with matching names (e.g., for my_dataset.tsv the corresponding metadata file is called my_dataset-metadata.json). The zip files must use the .zip extension.

    - is_repertoire (bool): If True, this imports a RepertoireDataset. If False, it imports a SequenceDataset or ReceptorDataset. By default, is_repertoire is set to True.

    - label_columns (list): For Sequence- or ReceptorDataset, this parameter can be used to explicitly set the column names of labels to import. These labels can be used as prediction target. When label_columns are not set, label names are attempted to be discovered automatically (any column name which is not used in the column_mapping). For RepertoireDataset labels, label_columns is ignored, metadata is discovered automatically from the metadata json.

    - paired (str): Required for Sequence- or ReceptorDatasets. This parameter determines whether to import a SequenceDataset (paired = False) or a ReceptorDataset (paired = True). In a ReceptorDataset, two sequences with chain types specified by receptor_chains are paired together based on the identifier given in the AIRR column named 'cell_id'.

    - receptor_chains (str): Required for ReceptorDatasets. Determines which pair of chains to import for each Receptor. Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum. If receptor_chains is not provided, the chain pair is automatically detected (only one chain pair type allowed per repertoire).

    - import_productive (bool): Whether productive sequences (with value 'T' in column productive) should be included in the imported sequences. By default, import_productive is True.

    - import_with_stop_codon (bool): Whether sequences with stop codons (with value 'T' in column stop_codon) should be included in the imported sequences. This only applies if column stop_codon is present. By default, import_with_stop_codon is False.

    - import_out_of_frame (bool): Whether out of frame sequences (with value 'F' in column vj_in_frame) should be included in the imported sequences. This only applies if column vj_in_frame is present. By default, import_out_of_frame is False.

    - import_illegal_characters (bool): Whether to import sequences that contain illegal characters, i.e., characters that do not appear in the sequence alphabet (amino acids including stop codon '*', or nucleotides). When set to false, filtering is only applied to the sequence type of interest (when running immuneML in amino acid mode, only entries with illegal characters in the amino acid sequence are removed). By default import_illegal_characters is False.

    - import_empty_nt_sequences (bool): imports sequences which have an empty nucleotide sequence field; can be True or False. By default, import_empty_nt_sequences is set to True.

    - import_empty_aa_sequences (bool): imports sequences which have an empty amino acid sequence field; can be True or False; for analysis on amino acid sequences, this parameter should be False (import only non-empty amino acid sequences). By default, import_empty_aa_sequences is set to False.

    - region_type (str): Which part of the sequence to import. By default, this value is set to IMGT_CDR3. This means the first and last amino acids are removed from the CDR3 sequence, as AIRR uses the IMGT junction. Specifying any other value will result in importing the sequences as they are. Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.

    - separator (str): Column separator, for AIRR this is by default "\\t".


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            datasets:
                my_airr_dataset:
                    format: IReceptor
                    params:
                        path: path/to/zipfiles/
                        is_repertoire: True # whether to import a RepertoireDataset
                        metadata_column_mapping: # metadata column mapping AIRR: immuneML for Sequence- or ReceptorDatasetDataset
                            airr_column_name1: metadata_label1
                            airr_column_name2: metadata_label2
                        import_productive: True # whether to include productive sequences in the dataset
                        import_with_stop_codon: False # whether to include sequences with stop codon in the dataset
                        import_out_of_frame: False # whether to include out of frame sequences in the dataset
                        import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
                        import_empty_nt_sequences: True # keep sequences even if the `sequences` column is empty (provided that other fields are as specified here)
                        import_empty_aa_sequences: False # remove all sequences with empty `sequence_aas` column
                        # Optional fields with AIRR-specific defaults, only change when different behavior is required:
                        separator: "\\t" # column separator
                        region_type: IMGT_CDR3 # what part of the sequence to import

    """
    REPERTOIRES_FOLDER = "repertoires/"

    def import_repertoire_dataset(self) -> RepertoireDataset:
        base_result_path = self.params.result_path / "tmp_airr"
        metadata_file_path = base_result_path / "metadata.csv"

        IReceptorImport._create_airr_repertoiredataset(self.params.path, base_result_path, metadata_file_path)

        self.params.path = base_result_path
        self.params.metadata_file = metadata_file_path

        dataset = super().import_repertoire_dataset()

        shutil.rmtree(base_result_path)

        return dataset

    def import_sequence_dataset(self):
        base_result_path = self.params.result_path / "tmp_airr"

        unzipped_path = base_result_path / "tmp_unzipped"
        IReceptorImport._unzip_files(self.params.path, unzipped_path, unzip_metadata=False)
        self.params.path = unzipped_path

        dataset = super().import_sequence_dataset()

        shutil.rmtree(unzipped_path)

        return dataset

    @staticmethod
    def _create_airr_repertoiredataset(input_zips_path: Path, base_result_path: Path, metadata_file_path: Path):
        unzipped_path = base_result_path / "tmp_unzipped/"
        PathBuilder.build(base_result_path / IReceptorImport.REPERTOIRES_FOLDER)

        IReceptorImport._unzip_files(input_zips_path, unzipped_path)

        all_metadata_dfs = []

        for airr_filename in unzipped_path.glob("*.tsv"):
            metadata_filename = unzipped_path / f"{airr_filename.stem}-metadata.json"

            sub_metadata_df = IReceptorImport._create_metadata_df(metadata_filename)
            files_written = IReceptorImport._split_airr_files(airr_filename, sub_metadata_df, base_result_path)
            sub_metadata_df = sub_metadata_df[files_written]

            all_metadata_dfs.append(sub_metadata_df)

        metadata_df = pd.concat(all_metadata_dfs, join="outer", ignore_index=True)
        metadata_df.fillna("NA", inplace=True)
        metadata_df.to_csv(metadata_file_path, index=False)

        shutil.rmtree(unzipped_path)


    @staticmethod
    def _unzip_files(path: Path, unzipped_path: Path, unzip_metadata=True):
        for zip_filename in path.glob("*.zip"):
            with zipfile.ZipFile(zip_filename, "r") as zip_object:
                for file in zip_object.filelist:
                    file.filename = f"{zip_filename.stem}_{file.filename}"
                    if file.filename.endswith(".tsv") or (file.filename.endswith("-metadata.json") and unzip_metadata):
                        zip_object.extract(file, path=unzipped_path)

    @staticmethod
    def _safe_get_field(dict, nested_fields):
        try:
            result = dict
            for field_name in nested_fields:
                result = result[field_name]
        except KeyError:
            result = None

        return result

    @staticmethod
    def _get_metadata_row(repertoire, sample, data_processing):
        repertoire_id = repertoire['repertoire_id']
        sample_processing_id = sample['sample_processing_id']
        data_processing_id = data_processing['data_processing_id']
        filename = f"{IReceptorImport.REPERTOIRES_FOLDER}{repertoire_id}_{sample_processing_id}_{data_processing_id}.tsv".replace(" ", "-")
        subject_id = repertoire["subject"]["subject_id"]

        study_id = IReceptorImport._safe_get_field(repertoire, ["study", "study_id"])
        species_label = IReceptorImport._safe_get_field(repertoire, ["subject", "species", "label"])
        organism_label = IReceptorImport._safe_get_field(repertoire, ["subject", "organism", "label"])
        sex = IReceptorImport._safe_get_field(repertoire, ["subject", "sex"])
        age_min = IReceptorImport._safe_get_field(repertoire, ["subject", "age_min"])
        age_max = IReceptorImport._safe_get_field(repertoire, ["subject", "age_max"])
        age_event = IReceptorImport._safe_get_field(repertoire, ["subject", "age_event"])
        ancestry_population = IReceptorImport._safe_get_field(repertoire, ["subject", "ancestry_population"])
        ethnicity = IReceptorImport._safe_get_field(repertoire, ["subject", "ethnicity"])
        race = IReceptorImport._safe_get_field(repertoire, ["subject", "race"])
        strain_name = IReceptorImport._safe_get_field(repertoire, ["subject", "strain_name"])

        tissue_label = IReceptorImport._safe_get_field(sample, ["tissue", "label"])
        disease_state_sample = IReceptorImport._safe_get_field(sample, ["disease_state_sample"])
        collection_time_point_relative = IReceptorImport._safe_get_field(sample, ["collection_time_point_relative"])
        collection_time_point_reference = IReceptorImport._safe_get_field(sample, ["collection_time_point_reference"])

        return (filename, subject_id, repertoire_id, sample_processing_id, data_processing_id, study_id, species_label,
                organism_label, sex, age_min, age_max, age_event, ancestry_population, ethnicity,
                race, strain_name, tissue_label, disease_state_sample, collection_time_point_relative,
                collection_time_point_reference)

    @staticmethod
    def _get_static_metadata_df(metadata_dict):
        identifiers = [IReceptorImport._get_metadata_row(repertoire, sample, data_processing)
                       for repertoire in metadata_dict["Repertoire"]
                       for sample in repertoire['sample']
                       for data_processing in repertoire['data_processing']]

        metadata_df = pd.DataFrame(identifiers,
                                   columns=["filename", "subject_id", "repertoire_id", "sample_processing_id",
                                            "data_processing_id", "study_id",
                                            "species_label", "organism_label", "sex", "age_min", "age_max", "age_event",
                                            "ancestry_population", "ethnicity", "race", "strain_name", "tissue_label",
                                            "disease_state_sample", "collection_time_point_relative",
                                            "collection_time_point_reference"])

        metadata_df.dropna(axis=1, how="all", inplace=True)

        return metadata_df

    @staticmethod
    def _add_diagnosis_columns(metadata_df, metadata_dict):
        unique_diseases = set(
            [str(diagnosis["disease_diagnosis"]["label"]) for repertoire in metadata_dict["Repertoire"] for diagnosis in
             repertoire['subject']['diagnosis']])

        id_sorted_repertoires = {repertoire["repertoire_id"]: repertoire for repertoire in metadata_dict["Repertoire"]}

        for disease_diagnosis_label in unique_diseases:
            corrected_label = disease_diagnosis_label.replace(" ", "_")
            metadata_df[corrected_label] = "NA"
            metadata_df[f"{corrected_label}_length"] = None
            metadata_df[f"{corrected_label}_stage"] = None
            metadata_df[f"{corrected_label}_immunogen"] = None

            for repertoire_id in metadata_df["repertoire_id"].unique():
                label_sorted_diagnoses = {str(diagnosis["disease_diagnosis"]["label"]): diagnosis for diagnosis in
                                            id_sorted_repertoires[repertoire_id]["subject"]["diagnosis"]}

                for current_diagnosis_label in label_sorted_diagnoses.keys():
                    if current_diagnosis_label == disease_diagnosis_label:
                        metadata_df.loc[metadata_df["repertoire_id"] == repertoire_id, corrected_label] = \
                        IReceptorImport._safe_get_field(label_sorted_diagnoses, [current_diagnosis_label, "study_group_description"])

                        metadata_df.loc[metadata_df["repertoire_id"] == repertoire_id, f"{corrected_label}_length"] = \
                        IReceptorImport._safe_get_field(label_sorted_diagnoses, [current_diagnosis_label, "disease_length"])

                        metadata_df.loc[metadata_df["repertoire_id"] == repertoire_id, f"{corrected_label}_stage"] = \
                        IReceptorImport._safe_get_field(label_sorted_diagnoses, [current_diagnosis_label, "disease_stage"])

                        metadata_df.loc[metadata_df["repertoire_id"] == repertoire_id, f"{corrected_label}_immunogen"] = \
                        IReceptorImport._safe_get_field(label_sorted_diagnoses, [current_diagnosis_label, "immunogen"])

        metadata_df.dropna(axis=1, how="all", inplace=True)

        return metadata_df

    @staticmethod
    def _create_metadata_df(metadata_json):
        with open(metadata_json) as json_file:
            metadata_dict = json.load(json_file)

        metadata_df = IReceptorImport._get_static_metadata_df(metadata_dict)
        metadata_df = IReceptorImport._add_diagnosis_columns(metadata_df, metadata_dict)

        return metadata_df

    @staticmethod
    def _split_airr_files(airr_file: Path, metadata_df: pd.DataFrame, result_path: Path):
        airr_df = airr.load_rearrangement(airr_file)
        files_written = []

        for filename, repertoire_id, sample_processing_id, data_processing_id in metadata_df[
            ["filename", "repertoire_id", "sample_processing_id", "data_processing_id"]].itertuples(index=False):

            subset = airr_df[airr_df["repertoire_id"] == repertoire_id]

            if "sample_processing_id" in subset.columns and any(subset["sample_processing_id"].str.len() > 0):
                subset = subset[subset["sample_processing_id"] == str(sample_processing_id)]
            if "data_processing_id" in subset.columns and any(subset["data_processing_id"].str.len() > 0):
                subset = subset[subset["data_processing_id"] == str(data_processing_id)]

            if subset.empty:
                files_written.append(False)
            else:
                subset.to_csv(result_path / filename, index=False, sep="\t")
                files_written.append(True)

        return files_written


    @staticmethod
    def get_documentation():
        doc = str(IReceptorImport.__doc__)

        chain_pair_values = str([chain_pair.name for chain_pair in ChainPair])[1:-1].replace("'", "`")
        region_type_values = str([region_type.name for region_type in RegionType])[1:-1].replace("'", "`")

        mapping = {
            "Valid values for receptor_chains are the names of the :py:obj:`~immuneML.data_model.receptor.ChainPair.ChainPair` enum.": f"Valid values are {chain_pair_values}.",
            "Valid values for region_type are the names of the :py:obj:`~immuneML.data_model.receptor.RegionType.RegionType` enum.": f"Valid values are {region_type_values}.",
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
