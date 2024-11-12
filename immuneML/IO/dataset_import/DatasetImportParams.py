from dataclasses import dataclass
from pathlib import Path

from immuneML.data_model.SequenceParams import ChainPair, RegionType
from immuneML.util.ParameterValidator import ParameterValidator


@dataclass
class DatasetImportParams:
    path: Path = None
    is_repertoire: bool = None
    metadata_file: Path = None
    paired: bool = None
    receptor_chains: ChainPair = None
    result_path: Path = None
    columns_to_load: list = None
    label_columns: list = None
    separator: str = None
    column_mapping: dict = None
    column_mapping_synonyms: dict = None
    region_type: RegionType = None
    import_productive: bool = None
    import_unknown_productivity: bool = None
    import_unproductive: bool = None
    import_with_stop_codon: bool = None
    import_out_of_frame: bool = None
    import_illegal_characters: bool = None
    number_of_processes: int = 1
    sequence_file_size: int = 50000
    organism: str = None
    import_empty_nt_sequences: bool = None
    import_empty_aa_sequences: bool = None
    dataset_file: Path = None

    @classmethod
    def build_object(cls, path: Path = None, metadata_file: Path = None, result_path: Path = None,
                     region_type: str = None, receptor_chains: str = None, **kwargs):
        params = {
            "path": Path(path) if path is not None else None,
            "metadata_file": Path(metadata_file) if metadata_file is not None else None,
            "result_path": Path(result_path) if result_path is not None else None,
            "region_type": RegionType[region_type.upper()] if region_type else None,
            "receptor_chains": ChainPair[receptor_chains.upper()] if receptor_chains else None,
        }

        if "column_mapping" in kwargs and kwargs['column_mapping']:
            ParameterValidator.assert_type_and_value(kwargs['column_mapping'], dict, cls.__name__, 'column_mapping')

            if not all(isinstance(el, int) for el in kwargs['column_mapping'].keys()):
                assert len(set(kwargs['column_mapping'].values())) == len(list(kwargs['column_mapping'].values())), \
                    (f"{cls.__name__}: Columns must be mapped to unique names, got: "
                     f"{list(kwargs['column_mapping'].values())}.")

        if kwargs.get('columns_to_load', None):
            ParameterValidator.assert_type_and_value(kwargs['columns_to_load'], list, cls.__name__, "columns_to_load")

        if kwargs.get('label_columns', None):
            ParameterValidator.assert_type_and_value(kwargs['label_columns'], list, cls.__name__, "label_columns")

        if kwargs.get('columns_to_load', None) and kwargs.get('label_columns', None):
            assert all(col_name in kwargs['columns_to_load'] for col_name in kwargs['label_columns']), \
                (f"{cls.__name__}: Some column names defined under 'label_columns' were not listed in 'columns_to_load'.\n"
                 f"label_columns: {kwargs['label_columns']}\n"
                 f"columns_to_load: {kwargs['columns_to_load']}\n"
                 f"To prevent this error, please add all label columns to columns_to_load.")

        if kwargs.get('columns_to_load', None) and kwargs.get('column_mapping', None):
            assert all(key in kwargs['columns_to_load'] for key in kwargs['column_mapping']), \
                f"{cls.__name__}: Some keys defined under 'column_mapping' were not listed in 'columns_to_load'."

            total_specified_fields = list(kwargs['column_mapping'].keys()) + kwargs['columns_to_load']
            if kwargs.get('label_columns', None):
                total_specified_fields += kwargs['label_columns']

            total_specified_fields = set(total_specified_fields)

            number_specified_fields = len(total_specified_fields)
            number_imported_fields = len(kwargs['columns_to_load'])
            assert number_specified_fields == number_imported_fields, \
                (f"{cls.__name__}: 'column_mapping', 'columns_to_load' and 'label_columns' fields were not correctly specified: "
                 f"{total_specified_fields} fields specified, and {number_imported_fields} fields are imported.\n"
                 f"All specified fields are: {sorted(total_specified_fields)}\n"
                 f"This does not match columns_to_load: {sorted(kwargs['columns_to_load'])}")

        params = {**kwargs, **params}
        return DatasetImportParams(**params)
