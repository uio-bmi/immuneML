import pandas as pd

from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata


class MetadataImport:

    @staticmethod
    def import_metadata(path) -> list:

        metadata_df = pd.read_csv(path)
        custom_keys = metadata_df.keys().values.tolist()

        standard_keys = ["filename"]
        for key in standard_keys:
            custom_keys.remove(key)

        mapping = metadata_df.apply(MetadataImport.extract_repertoire, axis=1, args=(custom_keys, )).values

        return mapping

    @staticmethod
    def extract_repertoire(row, custom_keys):

        custom_params = {key: row[key] for key in custom_keys}

        return {
            "rep_file": row["filename"],
            "metadata": RepertoireMetadata(custom_params=custom_params)
        }
