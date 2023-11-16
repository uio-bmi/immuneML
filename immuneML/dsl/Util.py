from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


def parse_exporters(instruction, location):
    if instruction["export_formats"] is not None:
        class_path = "dataset_export/"
        ParameterValidator.assert_all_in_valid_list(instruction["export_formats"],
                                                    ReflectionHandler.all_nonabstract_subclass_basic_names(DataExporter, 'Exporter', class_path),
                                                    location=location, parameter_name="export_formats")
        exporters = [ReflectionHandler.get_class_by_name(f"{item}Exporter", class_path) for item in instruction["export_formats"]]
    else:
        exporters = None

    return exporters
