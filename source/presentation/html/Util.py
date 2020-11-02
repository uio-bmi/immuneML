import glob
import os
import shutil
from enum import Enum

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
                return os.path.relpath(obj_abs_path, base_abs_path)
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
    def get_table_string_from_csv(csv_path: str, separator: str = ",", has_header: bool = True) -> str:
        table_string = "<table>\n"
        with open(csv_path, "r") as file:
            for index, line in enumerate(file.readlines()):
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
