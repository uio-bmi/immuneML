import logging
import os
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.StringHelper import StringHelper


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
        zip_file_path = shutil.make_archive(base_name=str(base_path / f"zip/{filename}"), format="zip", root_dir=str(path_to_zip))
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
    def get_logfile_path(base_path) -> str:
        path_str = ""

        logfile_path = list(base_path.glob("../*log.txt"))

        if len(logfile_path) == 1:
            path_str = os.path.relpath(logfile_path[0], base_path)
        else:
            try:
                logger = logging.getLogger()
                if len(logger.handlers) == 1:
                    handler = logger.handlers[0]
                    if isinstance(handler, logging.handlers.FileHandler):
                        path_str = os.path.relpath(str(handler.baseFilename), base_path)
            except Exception:
                pass

        return path_str

    @staticmethod
    def get_table_string_from_csv(csv_path: Path, separator: str = ",", has_header: bool = True) -> str:
        return Util.get_table_from_dataframe(pd.read_csv(csv_path, sep=separator, header=0 if has_header else None))

    @staticmethod
    def get_table_from_dataframe(df: pd.DataFrame):
        return df.to_html(index_names=False, index=False).replace('<table border="1" class="dataframe">', '<table>')

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
                        logging.warning(f"HTML util: one of the report outputs was not returned properly from the "
                                        f"report {report_result.name}, and it will not be moved to HTML output folder.")

        return report_result

    @staticmethod
    def get_dataset_summary_info(dataset, label_name: str = None) -> dict:
        """
        Returns a small summary of a dataset (number of examples, and if a label is given and available for that
        dataset, the number of examples per class for that label), to be shown as a quick overview at the top of
        an HTML report page.
        """
        example_count = dataset.get_example_count()
        class_counts_str = None

        if label_name is not None and label_name in dataset.get_label_names():
            try:
                label_values = dataset.get_metadata([label_name], return_df=True)[label_name]
                counts = label_values.value_counts(dropna=False).sort_index(key=lambda index: index.astype(str))
                class_counts_str = ", ".join(f"{class_name}: {count}" for class_name, count in counts.items())
            except Exception as e:
                logging.warning(f"Util: could not compute the number of examples per class for label "
                                f"{label_name} for dataset {dataset.name}: {e}")

        return {"example_count": example_count, "class_counts_str": class_counts_str,
                "has_class_counts": bool(class_counts_str)}

    @staticmethod
    def make_dataset_html_map(dataset, dataset_key="dataset"):
        return {f"{dataset_key}_name": dataset.name,
                f"{dataset_key}_type": StringHelper.camel_case_to_word_string(type(dataset).__name__),
                f"{dataset_key}_locus": ", ".join(dataset.get_locus()),
                f"{dataset_key}_size": f"{dataset.get_example_count()} {type(dataset).__name__.replace('Dataset', 's').lower()}",
                f"{dataset_key}_labels": [{f"{dataset_key}_label_name": label_name,
                                           f"{dataset_key}_label_classes": get_label_values_to_show(dataset.labels.get(label_name, []), dataset.get_example_count())}
                                                for label_name in dataset.get_label_names()],
                f"show_{dataset_key}_labels": len(dataset.get_label_names()) > 0}

def get_label_values_to_show(label_values: list, dataset_size: int):
    if isinstance(label_values, list):
        if len(label_values) == dataset_size:
            return "unique per example"
        elif any(isinstance(label, float) and not label != np.nan for label in label_values):
            try:
                return f"{min(label_values):.2f} to {max(label_values):.2f}"
            except Exception as e:
                return 'mixed value types'
        elif any(isinstance(label, int) for label in label_values):
            try:
                return f"{min(label_values)} to {max(label_values)}"
            except Exception as e:
                return 'mixed value types'
        else:
            return ", ".join(str(label) for label in label_values)
    else:
        return label_values