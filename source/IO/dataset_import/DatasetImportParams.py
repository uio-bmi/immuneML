from dataclasses import dataclass

from source.data_model.receptor.ChainPair import ChainPair
from source.data_model.receptor.RegionDefinition import RegionDefinition
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
    region_definition: RegionDefinition = None
    file_size: int = None
    import_productive: bool = None
    import_with_stop_codon: bool = None
    import_out_of_frame: bool = None
    paired: bool = None
    chains: ChainPair = None
    batch_size: int = 1
    sequence_file_size: int = 50000

    @classmethod
    def build_object(cls, region_type: str = None, region_definition: str = None, chains: str = None, **kwargs):
        params = {
            "region_type": RegionType[region_type.upper()] if region_type else None,
            "region_definition": RegionDefinition[region_definition.upper()] if region_definition else None,
            "chains": ChainPair[chains.upper()] if chains else None
        }
        params = {**kwargs, **params}
        return DatasetImportParams(**params)

# OLD:
    # metadata_file: str = None
    # path: str = None
    # result_path: str = None
    # batch_size: int = 1
    # import_productive: bool = None
    # import_with_stop_codon: bool = None
    # import_out_of_frame: bool = None
    # columns_to_load: list = None
    # separator: str = None
    # column_mapping: dict = None
    # region_type: RegionType = None
    # region_definition: RegionDefinition = None
    # file_size: int = None
    # paired: bool = None