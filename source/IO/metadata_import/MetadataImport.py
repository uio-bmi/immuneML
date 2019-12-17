import pandas as pd


class MetadataImport:

    @staticmethod
    def import_metadata(path) -> list:

        metadata_df = pd.read_csv(path)
        custom_keys = metadata_df.keys().values.tolist()

        custom_keys.remove("filename")
        custom_keys.remove("donor")

        mapping = metadata_df.apply(MetadataImport.extract_repertoire, axis=1, args=(custom_keys, )).values

        return mapping

    @staticmethod
    def extract_repertoire(row, custom_keys):

        custom_params = {key: row[key] for key in custom_keys}

        return {
            "rep_file": row["filename"],
            "donor": row["donor"],
            "metadata": custom_params
        }
