import copy
import os
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.DataLoader import DataLoader
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.IO.metadata_import.MetadataImport import MetadataImport
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.Constants import Constants
from source.util.PathBuilder import PathBuilder


class GenericLoader(DataLoader):

    def load(self, path, params: dict = None) -> RepertoireDataset:
        params = copy.deepcopy(params)
        if os.path.exists(params["result_path"] + "/dataset.pkl"):
            return PickleLoader().load(params["result_path"] + "/dataset.pkl")
        PathBuilder.build(params["result_path"])
        metadata = MetadataImport.import_metadata(params["metadata_file"])
        params["metadata"] = metadata
        filepaths = [path + "/" + rep["rep_file"] for rep in metadata]
        dataset = self._load(filepaths, params)
        PickleExporter.export(dataset, params["result_path"], "dataset.pkl")
        return dataset

    def _load(self, filepaths: list, params: dict) -> RepertoireDataset:

        arguments = [(index, filepath, params) for index, filepath in enumerate(filepaths)]

        with Pool(params.get("batch_size", 1)) as pool:
            output = pool.starmap(self._load_repertoire, arguments)

        output = [out for out in output if out != (None, None)]

        repertoires = [out[0] for out in output]
        custom_params = self._prepare_custom_params([out[1] for out in output])

        dataset = RepertoireDataset(repertoires=repertoires, params=custom_params, metadata_file=params["metadata_file"])
        return dataset

    def _build_dtype(self, params, column_mapping):

        dtype_default = {
            "v_subgroup": str, "v_genes": str, "v_allele": str, "j_subgroup": str, "j_genes": str, "j_allele": str,
            "sequence_aas": str, "sequences": str, "templates": int, "frame_type": str}

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
            df['sequence_aas'] = df["sequence_aas"].str[1:-1]

        df = df.replace(["unresolved", "no data", "na", "unknown", "null", "nan", np.nan], Constants.UNKNOWN)

        return df

    def get_sequences_from_df(self, df, params):

        column_mapping = params.get("column_mapping", {})

        df = df.rename(columns={j: i for i, j in column_mapping.items()})

        return df.apply(self._load_sequence, axis=1, args=(params,)).values

    def _load_repertoire_from_file(self, filepath, params, identifier) -> dict:
        df = self._read_preprocess_file(filepath, params)

        sequence_lists = self.get_sequence_lists_from_df(df, params)
        metadata = self._extract_repertoire_metadata(filepath, params, df)

        repertoire_inputs = {**{"metadata": metadata, "identifier": identifier, "path": params["result_path"]},
                             **sequence_lists}

        repertoire = SequenceRepertoire.build(**repertoire_inputs)

        return repertoire

    def get_sequence_lists_from_df(self, df, params):

        standard_fields = {
            field: df[field].values if field in df.columns and df[field] is not None else None for field in SequenceRepertoire.FIELDS
        }

        if "additional_columns" in params:
            custom_fields = {field: df[field].values if field in df.columns and df[field] is not None else None
                             for field in params["additional_columns"]}
        else:
            custom_fields = {}

        return {**standard_fields, **{"custom_lists": custom_fields}}

    def _load_repertoire(self, index, filepath, params):
        identifier = str(params["metadata"][index]["donor"]) if "metadata" in params else str(os.path.basename(filepath).rpartition(".")[0])
        metadata_filename = f"{params['result_path']}{identifier}_metadata.pickle"
        data_filename = f"{params['result_path']}{identifier}.npy"
        if not os.path.exists(metadata_filename) or not os.path.exists(data_filename):
            repertoire = self._load_repertoire_from_file(filepath, params, identifier)
        else:
            repertoire = SequenceRepertoire(data_filename, metadata_filename, identifier)

        return repertoire, repertoire.metadata

    def _load_sequence(self, row, params) -> ReceptorSequence:

        metadata = SequenceMetadata(v_subgroup=row.get("v_subgroup", Constants.UNKNOWN),
                                    v_gene=row.get("v_gene", Constants.UNKNOWN),
                                    v_allele=row.get("v_allele", Constants.UNKNOWN),
                                    j_subgroup=row.get("j_subgroup", Constants.UNKNOWN),
                                    j_gene=row.get("j_gene", Constants.UNKNOWN),
                                    j_allele=row.get("j_allele", Constants.UNKNOWN),
                                    chain=row.get("chain", "TRB"),
                                    count=int(row.get("templates", "0")) if str(row.get("templates", "0")).isdigit() else 0,
                                    frame_type=row.get("frame_type", SequenceFrameType.IN.value),
                                    region_type=params.get("region_type", Constants.UNKNOWN))

        sequence = ReceptorSequence(amino_acid_sequence=row.get("amino_acid", None),
                                    nucleotide_sequence=row.get("nucleotide", None),
                                    metadata=metadata)

        for column in row.keys():
            if params.get("additional_columns") == "*" or column in params.get("additional_columns", []):
                metadata.custom_params[column] = row[column]

        return sequence

    def _extract_repertoire_metadata(self, filepath, params, df) -> dict:
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
                    custom_params[key].add(tuple(p[key]))
                else:
                    custom_params[key] = {tuple(p[key])}

        return custom_params
