from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.data_model.receptor.RegionType import RegionType


@dataclass
class DatasetImportParams:
    path: Path = None
    is_repertoire: bool = None
    metadata_file: Path = None
    paired: bool = None
    receptor_chains: ChainPair = None
    result_path: Path = None
    columns_to_load: list = None
    separator: str = None
    column_mapping: dict = None
    region_type: RegionType = None
    import_productive: bool = None
    import_unproductive: bool = None
    import_with_stop_codon: bool = None
    import_out_of_frame: bool = None
    import_illegal_characters: bool = None
    metadata_column_mapping: dict = None
    number_of_processes: int = 1
    sequence_file_size: int = 50000
    organism: str = None
    import_empty_nt_sequences: bool = None
    import_empty_aa_sequences: bool = None

    @classmethod
    def build_object(cls, path: Path = None, metadata_file: Path = None, result_path: Path = None, region_type: str = None, receptor_chains: str = None, **kwargs):
        params = {
            "path": Path(path) if path is not None else None,
            "metadata_file": Path(metadata_file) if metadata_file is not None else None,
            "result_path": Path(result_path) if result_path is not None else None,
            "region_type": RegionType[region_type.upper()] if region_type else None,
            "receptor_chains": ChainPair[receptor_chains.upper()] if receptor_chains else None,
        }
        params = {**kwargs, **params}
        return DatasetImportParams(**params)
