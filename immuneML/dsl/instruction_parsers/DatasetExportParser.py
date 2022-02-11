from pathlib import Path

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.dataset_generation.DatasetExportInstruction import DatasetExportInstruction


class DatasetExportParser:
    """
    Specification of instruction with a random datasets:

    definitions:
      datasets:
        my_generated_dataset: # a dataset to be exported in the given format
          format: RandomRepertoireDataset
          params:
            result_path: generated_dataset/
            repertoire_count: 100
            sequence_count_probabilities:
              100: 0.5
              120: 0.5
            sequence_length_probabilities:
              12: 0.333
              13: 0.333
              14: 0.333
            labels:
              immune_event_1:
                yes: 0.5
                no: 0.5
      preprocessing_sequences:
            my_preprocessing:
                - my_filter:
                    ClonesPerRepertoireFilter:
                        lower_limit: 110
                        upper_limit: 200
    instructions:
      my_instruction1: # instruction name
        type: DatasetExport
        datasets: # list of datasets to export
          - my_generated_dataset
        preprocessing_sequence: my_preprocessing_sequence
        number_of_processes: 4
        export_formats: # list of formats to export the datasets to
          - AIRR
          - ImmuneML

    """

    REQUIRED_KEYS = ["type", "datasets", "export_formats"]
    OPTIONAL_KEYS = ["preprocessing_sequence", "number_of_processes"]

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> DatasetExportInstruction:
        location = "DatasetExportParser"
        ParameterValidator.assert_keys(list(instruction.keys()), DatasetExportParser.REQUIRED_KEYS + DatasetExportParser.OPTIONAL_KEYS, location, key, False)
        ParameterValidator.assert_keys_present(list(instruction.keys()), DatasetExportParser.REQUIRED_KEYS, location, key)

        valid_formats = ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, "Exporter", 'dataset_export/')
        ParameterValidator.assert_all_in_valid_list(instruction["export_formats"], valid_formats, location, "export_formats")
        ParameterValidator.assert_all_in_valid_list(instruction["datasets"], symbol_table.get_keys_by_type(SymbolType.DATASET), location, "datasets")

        if "number_of_processes" in instruction:
            ParameterValidator.assert_type_and_value(instruction["number_of_processes"], int, location, "number_of_processes", 1)

        return DatasetExportInstruction(datasets=[symbol_table.get(dataset_key) for dataset_key in instruction["datasets"]],
                                        exporters=[ReflectionHandler.get_class_by_name(f"{key}Exporter", "dataset_export/")
                                                       for key in instruction["export_formats"]],
                                        number_of_processes=instruction["number_of_processes"] if "number_of_processes" in instruction else 1,
                                        preprocessing_sequence=symbol_table.get(instruction["preprocessing_sequence"]) if "preprocessing_sequence" in instruction else None,
                                        name=key)
