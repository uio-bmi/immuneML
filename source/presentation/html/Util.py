import glob
import logging
import os
import shutil
from enum import Enum
from pathlib import Path

from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.util.PathBuilder import PathBuilder


class Util:

    @staticmethod
    def to_dict_recursive(obj, base_path: Path):
        if not hasattr(obj, "__dict__"):
            if isinstance(obj, Path) and obj.is_file():
                obj_abs_path = obj.absolute()
                base_abs_path = base_path.absolute()
                res_path = Path(os.path.relpath(obj_abs_path, base_abs_path))
                return res_path
            elif isinstance(obj, str) and os.path.isfile(obj):
                assert False, "Update paths"
            elif hasattr(obj, "name"):
                return obj.name
            elif isinstance(obj, list):
                return [Util.to_dict_recursive(element, base_path) for element in obj]
            else:
                return obj if obj is not None else ""
        elif isinstance(obj, Enum):
            return str(obj)
        else:
            vars_obj = vars(obj)
            return {
                attribute_name: Util.to_dict_recursive(vars_obj[attribute_name], base_path) for attribute_name in vars_obj.keys()
            }

    @staticmethod
    def get_css_content(css_path: Path):
        with css_path.open("rb") as file:
            content = file.read()
        return content

    @staticmethod
    def make_downloadable_zip(base_path: Path, path_to_zip: Path, filename: str = "") -> Path:
        if filename == "":
            filename = Path("_".join(os.path.relpath(path_to_zip, base_path).replace(".", "").split("/")))

        PathBuilder.build(base_path / "zip")
        zip_file_path = Path(shutil.make_archive(base_name=base_path / f"zip/{filename}", format="zip", root_dir=path_to_zip))
        return zip_file_path

    @staticmethod
    def get_full_specs_path(base_path):
        specs_path = list(glob.glob(str(base_path / "../**/full*.yaml"), recursive=True))
        if len(specs_path) == 1:
            path_str = os.path.relpath(specs_path[0], base_path)
        else:
            path_str = ""

        return Path(path_str)

    @staticmethod
    def get_table_string_from_csv_string(csv_string: str, separator: str = ",", has_header: bool = True) -> str:
        table_string = "<table>\n"
        for index, line in enumerate(csv_string.splitlines()):
            if index == 0 and has_header:
                table_string += "<thead>\n"
            table_string += "<tr>\n"
            for col in line.split(separator):
                table_string += f"<td>{col}</td>\n"
            table_string += "</tr>\n"
            if index == 0 and has_header:
                table_string += "</thead>\n"
        table_string += "</table>\n"
        return table_string

    @staticmethod
    def get_table_string_from_csv(csv_path: Path, separator: str = ",", has_header: bool = True) -> str:
        with csv_path.open("r") as file:
            table_string = Util.get_table_string_from_csv_string(file.read())
        return table_string

    @staticmethod
    def update_report_paths(report_result: ReportResult, path: Path) -> ReportResult:
        for attribute in vars(report_result):
            attribute_value = getattr(report_result, attribute)
            if isinstance(attribute_value, list):
                for output in attribute_value:
                    if isinstance(output, ReportOutput): # todo fis this!! assert False filename is strange?
                        new_filename = os.path.relpath(path=output.path, start=path).replace("..", "").replace("/", "_")
                        new_filename = new_filename[1:] if new_filename[0] == '_' else new_filename # todo only first _ gets removed?
                        new_path = path / new_filename
                        if output.path != new_path:
                            shutil.copyfile(src=output.path, dst=new_path)
                            output.path = new_path
                    else:
                        logging.warning(f"HTML util: one of the report outputs was not returned properly from the report {report_result.name}, "
                                        f"and it will not be moved to HTML output folder.")

        return report_result
