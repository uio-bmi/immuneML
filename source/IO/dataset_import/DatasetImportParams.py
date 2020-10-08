from dataclasses import dataclass

from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.RegionType import RegionType


@dataclass
class DatasetImportParams:
    is_repertoire: bool = True
    path: str = None
    metadata_file: str = None
    result_path: str = None
    columns_to_load: list = None
    separator: str = None
    column_mapping: dict = None
    region_type: RegionType = None
    import_productive: bool = None
    import_unproductive: bool = None
    import_with_stop_codon: bool = None
    import_out_of_frame: bool = None
    paired: bool = None
    receptor_chains: ChainPair = None
    metadata_column_mapping: list = None
    batch_size: int = 1
    sequence_file_size: int = 50000


    @classmethod
    def build_object(cls, region_type: str = None, receptor_chains: str = None, **kwargs):
        params = {
            "region_type": RegionType[region_type.upper()] if region_type else None,
            "receptor_chains": ChainPair[receptor_chains.upper()] if receptor_chains else None
        }
        params = {**kwargs, **params}
        return DatasetImportParams(**params)
