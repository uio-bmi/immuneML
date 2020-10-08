from dataclasses import dataclass

from source.data_model.receptor.ChainPair import ChainPair


@dataclass
class IRISImportParams:
    is_repertoire: bool = True
    metadata_file: str = None
    path: str = None
    result_path: str = None
    sequence_file_size: int = None
    paired: bool = None
    receptor_chains: ChainPair = None
    import_dual_chains: bool = None
    import_all_gene_combinations: bool = None
    batch_size: int = 1
    separator: str = None
    extra_columns_to_load: list = None

    @classmethod
    def build_object(cls, receptor_chains: str = None, **kwargs):
        params = {
            "receptor_chains": ChainPair[receptor_chains.upper()] if receptor_chains else None
        }
        params = {**kwargs, **params}
        return IRISImportParams(**params)
