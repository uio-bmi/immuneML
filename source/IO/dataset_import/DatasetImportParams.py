from dataclasses import dataclass

from source.data_model.receptor.RegionDefinition import RegionDefinition
from source.data_model.receptor.RegionType import RegionType


@dataclass
class DatasetImportParams:
    metadata_file: str = None
    path: str = None
    result_path: str = None
    batch_size: int = 1
    import_productive: bool = None
    import_with_stop_codon: bool = None
    import_out_of_frame: bool = None
    columns_to_load: list = None
    separator: str = None
    column_mapping: dict = None
    region_type: RegionType = None
    region_definition: RegionDefinition = None
    file_size: int = None
    paired: bool = None
    misc: dict = None
