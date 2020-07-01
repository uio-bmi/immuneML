import glob
import os
import shutil

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
                return os.path.relpath(obj, base_path) + "/" if os.path.relpath(obj, base_path) != "" and os.path.isdir(obj) else os.path.relpath(obj, base_path)
            else:
                return obj if obj is not None else ""
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
    def make_downloadable_zip(base_path, tmp_path):
        filename = "_".join(os.path.relpath(tmp_path, base_path).replace(".", "").split("/"))[1:-1]
        PathBuilder.build(f"{base_path}zip/")
        zip_file_path = shutil.make_archive(f"{base_path}zip/{filename}", "zip", tmp_path)
        return os.path.relpath(zip_file_path, base_path)

    @staticmethod
    def get_full_specs_path(base_path, state_result_path):
        specs_path = list(glob.glob(f"{base_path}**/full*.yaml", recursive=True))
        if len(specs_path) == 1:
            if base_path == state_result_path:
                return os.path.relpath(specs_path[0], base_path)
            else:
                return os.path.relpath(specs_path[0], state_result_path)
        else:
            return ""
