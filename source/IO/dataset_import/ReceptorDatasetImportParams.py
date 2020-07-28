from dataclasses import dataclass

from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.RegionType import RegionType


@dataclass
class ReceptorDatasetImportParams:
    path: str = None
    result_path: str = None
    columns_to_load: list = None
    separator: str = None
    column_mapping: dict = None
    region_type: RegionType = None
    file_size: int = None
    paired: bool = None
    chains: ChainPair = None
    organism: str = None

    @classmethod
    def build_object(cls, region_type: str = None, chains: str = None, **kwargs):
        params = {
            "region_type": RegionType[region_type.upper()] if region_type else None,
            "chains": ChainPair[chains.upper()] if chains else None
        }
        params = {**kwargs, **params}
        return ReceptorDatasetImportParams(**params)
