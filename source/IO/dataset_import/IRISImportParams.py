from dataclasses import dataclass


@dataclass
class IRISImportParams:
    metadata_file: str = None
    path: str = None
    result_path: str = None
    file_size: int = None
    paired: bool = None
    import_dual_chains: bool = None
    import_all_gene_combinations: bool = None
    batch_size: int = 1
    separator: str = None
    extra_columns_to_load: list = None

    @classmethod
    def build_object(cls, **kwargs):
        return IRISImportParams(**kwargs)
