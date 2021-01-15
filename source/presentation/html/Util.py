import glob
import logging
import os
import shutil
from enum import Enum

from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.util.PathBuilder import PathBuilder


class Util:

    @staticmethod
    def to_dict_recursive(obj, base_path):
        if not hasattr(obj, "__dict__"):
            if hasattr(obj, "name"):
                return obj.name
            elif isinstance(obj, list):
                return [Util.to_dict_recursive(element, base_path) for element in obj]
            elif isinstance(obj, str) and os.path.isfile(obj):
                obj_abs_path = os.path.abspath(obj)
                base_abs_path = os.path.abspath(base_path)
                res_path = os.path.relpath(obj_abs_path, base_abs_path)
                return res_path
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
                if ".svg" in getattr(obj, "path", ""):
                    result['is_embed'] = False
                else:
                    result['is_embed'] = True
            return result

    @staticmethod
    def get_css_content(css_path: str):
        with open(css_path, "rb") as file:
            content = file.read()
        return content

    @staticmethod
    def make_downloadable_zip(base_path, path_to_zip, filename: str = ""):
        if filename == "":
            filename = "_".join(os.path.relpath(path_to_zip, base_path).replace(".", "").split("/"))

        PathBuilder.build(f"{base_path}zip/")
        zip_file_path = shutil.make_archive(base_name=f"{base_path}zip/{filename}", format="zip", root_dir=path_to_zip)
        return zip_file_path

    @staticmethod
    def get_full_specs_path(base_path):
        specs_path = list(glob.glob(f"{base_path}/../**/full*.yaml", recursive=True))
        if len(specs_path) == 1:
            return os.path.relpath(specs_path[0], base_path)
        else:
            return ""

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
    def get_table_string_from_csv(csv_path: str, separator: str = ",", has_header: bool = True) -> str:
        with open(csv_path, "r") as file:
            table_string = Util.get_table_string_from_csv_string(file.read())
        return table_string

    @staticmethod
    def update_report_paths(report_result: ReportResult, path: str) -> ReportResult:
        for attribute in vars(report_result):
            attribute_value = getattr(report_result, attribute)
            if isinstance(attribute_value, list):
                for output in attribute_value:
                    if isinstance(output, ReportOutput):
                        new_filename = os.path.relpath(path=output.path, start=path).replace("..", "").replace("/", "_")
                        new_filename = new_filename[1:] if new_filename[0] == '_' else new_filename
                        new_path = path + new_filename
                        if output.path != new_path:
                            shutil.copyfile(src=output.path, dst=new_path)
                            output.path = new_path
                    else:
                        logging.warning(f"HTML util: one of the report outputs was not returned properly from the report {report_result.name}, "
                                        f"and it will not be moved to HTML output folder.")

        return report_result
