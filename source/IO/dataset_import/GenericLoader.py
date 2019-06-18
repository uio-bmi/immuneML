import os
import pickle
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DataLoader import DataLoader
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.IO.metadata_import.MetadataImport import MetadataImport
from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireGenerator import RepertoireGenerator
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.environment.Constants import Constants
from source.util.PathBuilder import PathBuilder


class GenericLoader(DataLoader):

    def load(self, path, params: dict = None) -> Dataset:
        if os.path.exists(params["result_path"] + "/dataset.pkl"):
            return PickleLoader().load(params["result_path"] + "/dataset.pkl")
        PathBuilder.build(params["result_path"])
        metadata = MetadataImport.import_metadata(params["metadata_file"])
        params["metadata"] = metadata
        filepaths = [path + "/" + rep["rep_file"] for rep in metadata]
        dataset = self._load(filepaths, params)
        PickleExporter.export(dataset, params["result_path"], "dataset.pkl")
        return dataset

    def _load(self, filepaths: list, params: dict) -> Dataset:

        arguments = [(filepath, params) for filepath in filepaths]

        with Pool(params.get("batch_size", 1)) as pool:
            output = pool.starmap(self._load_repertoire, arguments)

        output = [out for out in output if out != (None, None)]

        repertoire_filenames = [out[0] for out in output]
        custom_params = self._prepare_custom_params([out[1] for out in output])

        dataset = Dataset(filenames=repertoire_filenames, params=custom_params, metadata_file=params["metadata_file"])
        return dataset

    def _build_dtype(self, params, column_mapping):

        dtype_default = {
            "v_subgroup": str, "v_gene": str, "v_allele": str, "j_subgroup": str, "j_gene": str, "j_allele": str,
            "amino_acid": str, "nucleotide": str, "templates": int, "frame_type": str}

        dtype = {custom: dtype_default.get(default, None) for default, custom in column_mapping.items()}

        column_dtype = params.get("additional_columns_dtype", {})

        return {**dtype, **column_dtype}

    def _build_column_mapping(self, params):
        return params.get("column_mapping", {
            "v_subgroup": "v_subgroup", "v_gene": "v_gene", "v_allele": "v_allele", "j_subgroup": "j_subgroup",
            "j_gene": "j_gene", "j_allele": "j_allele", "amino_acid": "amino_acid", "nucleotide": "nucleotide",
            "templates": "templates", "frame_type": "frame_type"})

    def _read_preprocess_file(self, filepath, params):

        column_mapping = self._build_column_mapping(params)
        dtype = self._build_dtype(params, column_mapping)
        usecols = None if params.get("additional_columns") == "*" else list(column_mapping.values()) + params.get("additional_columns", [])
        separator = params.get("separator", "\t")

        try:
            df = pd.read_csv(filepath, sep=separator, iterator=False, usecols=usecols, dtype=dtype)
        except:
            df = pd.read_csv(filepath, sep=separator, iterator=False, usecols=usecols, dtype=dtype, encoding="latin1")

        df = df.rename(columns={j: i for i, j in column_mapping.items()})

        if params.get("strip_CF", False):
            df['amino_acid'] = df["amino_acid"].str[1:-1]

        df = df.replace(["unresolved", "no data", "na", "unknown", "null", "nan", np.nan], Constants.UNKNOWN)

        return df

    def get_sequences_from_df(self, df, params):

        column_mapping = params.get("column_mapping", {})

        df = df.rename(columns={j: i for i, j in column_mapping.items()})

        return df.apply(self._load_sequence, axis=1, args=(params,)).values

    def _load_repertoire_from_file(self, filepath, params, repertoire_filename, identifier) -> RepertoireMetadata:
        df = self._read_preprocess_file(filepath, params)

        sequences = self.get_sequences_from_df(df, params)
        metadata = self._extract_repertoire_metadata(filepath, params, df)

        del df

        repertoire = Repertoire(sequences=sequences, metadata=metadata, identifier=identifier)

        with open(repertoire_filename, "wb") as f:
            pickle.dump(repertoire, f)

        del sequences
        del repertoire

        return metadata

    def _load_repertoire(self, filepath, params):
        identifier = str(os.path.basename(filepath).rpartition(".")[0])
        repertoire_filename = params["result_path"] + identifier + ".pickle"

        if os.path.exists(repertoire_filename):
            repertoire = RepertoireGenerator.load_repertoire(repertoire_filename)
            metadata = repertoire.metadata
            del repertoire
        else:
            metadata = self._load_repertoire_from_file(filepath, params, repertoire_filename, identifier)

        custom_params = metadata.custom_params if metadata is not None else {}

        return repertoire_filename, custom_params

    def _load_sequence(self, row, params) -> ReceptorSequence:

        metadata = SequenceMetadata(v_subgroup=row.get("v_subgroup", Constants.UNKNOWN),
                                    v_gene=row.get("v_gene", Constants.UNKNOWN),
                                    v_allele=row.get("v_allele", Constants.UNKNOWN),
                                    j_subgroup=row.get("j_subgroup", Constants.UNKNOWN),
                                    j_gene=row.get("j_gene", Constants.UNKNOWN),
                                    j_allele=row.get("j_allele", Constants.UNKNOWN),
                                    chain=row.get("chain", "TRB"),
                                    count=row.get("templates", 0),
                                    frame_type=row.get("frame_type", SequenceFrameType.IN.value),
                                    region_type=params.get("region_type", Constants.UNKNOWN))

        sequence = ReceptorSequence(amino_acid_sequence=row.get("amino_acid", None),
                                    nucleotide_sequence=row.get("nucleotide", None),
                                    metadata=metadata)

        for column in row.keys():
            if params.get("additional_columns") == "*" or column in params.get("additional_columns", []):
                metadata.custom_params[column] = row[column]

        return sequence

    def _extract_repertoire_metadata(self, filepath, params, df) -> RepertoireMetadata:
        if "metadata" in params:
            metadata = [m for m in params["metadata"] if os.path.basename(m["rep_file"]) == os.path.basename(filepath)][0]["metadata"]
        else:
            metadata = None
        return metadata

    def _prepare_custom_params(self, params: list) -> dict:
        custom_params = {}
        for p in params:
            for key in p.keys():
                if key in custom_params:
                    custom_params[key].add(p[key])
                else:
                    custom_params[key] = {p[key]}

        return custom_params
