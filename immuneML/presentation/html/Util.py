import logging
import os
import shutil
from enum import Enum
from pathlib import Path
import numpy as np

from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder


class Util:

    @staticmethod
    def to_dict_recursive(obj, base_path: Path):
        if not hasattr(obj, "__dict__"):
            if (isinstance(obj, Path) or isinstance(obj, str)) and os.path.isfile(obj):
                obj_abs_path = os.path.abspath(obj)
                base_abs_path = os.path.abspath(str(base_path))
                res_path = os.path.relpath(obj_abs_path, base_abs_path)
                return res_path
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
            result = {
                attribute_name: Util.to_dict_recursive(vars_obj[attribute_name], base_path) for attribute_name in vars_obj.keys()
            }
            if isinstance(obj, ReportOutput):
                path = getattr(obj, "path")

                if path is not None:
                    result['is_download_link'] = True
                    result['download_link'] = os.path.relpath(path=getattr(obj, "path"), start=base_path)
                    result['file_name'] = getattr(obj, "name")

                    if any([ext == path.suffix for ext in ['.svg', '.jpg', '.png']]):
                        result['is_embed'] = False
                    else:
                        result['is_embed'] = True
                else:
                    result['is_download_link'] = False

            return result

    @staticmethod
    def get_css_content(css_path: Path):
        with css_path.open("rb") as file:
            content = file.read()
        return content

    @staticmethod
    def make_downloadable_zip(base_path: Path, path_to_zip: Path, filename: str = "") -> str:
        if filename == "":
            filename = "_".join(Path(os.path.relpath(str(path_to_zip), str(base_path)).replace(".", "")).parts)

        PathBuilder.build(base_path / "zip")
        zip_file_path = shutil.make_archive(base_name=base_path / f"zip/{filename}", format="zip", root_dir=str(path_to_zip))
        return zip_file_path

    @staticmethod
    def get_full_specs_path(base_path) -> str:
        specs_path = list(base_path.glob("../**/full*.yaml"))

        if len(specs_path) == 1:
            path_str = os.path.relpath(specs_path[0], base_path)
        else:
            path_str = ""

        return path_str

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
                    if isinstance(output, ReportOutput):
                        new_filename = "_".join([part for part in Path(os.path.relpath(path=str(output.path), start=str(path))).parts if part != ".."])
                        new_path = path / new_filename

                        if output.path != new_path:
                            shutil.copyfile(src=str(output.path), dst=new_path)
                            output.path = new_path
                    else:
                        logging.warning(f"HTML util: one of the report outputs was not returned properly from the report {report_result.name}, "
                                        f"and it will not be moved to HTML output folder.")

        return report_result
