import glob
import importlib
import os

from source.environment.EnvironmentSettings import EnvironmentSettings


class ReportParser:

    @staticmethod
    def parse_reports(reports):

        if isinstance(reports, list):
            keys = reports
            params = None
        else:
            keys = reports.keys()
            params = reports

        result = ReportParser._parse_by_key(keys, params)
        return result

    @staticmethod
    def _import_all_report_classes() -> dict:
        report_classes = {}

        # import all report classes
        for name in glob.glob(EnvironmentSettings.root_path + "source/reports/**/*.py"):
            if "__init__" not in name:
                mod = importlib.import_module(name[name.rfind("source"):-3].replace("../", "").replace("/", "."))
                tmp = getattr(mod, os.path.basename(name)[:-3])
                if len(tmp.__abstractmethods__) == 0:
                    report_classes[os.path.basename(name)[:-3]] = tmp

        return report_classes

    @staticmethod
    def _parse_by_key(keys, params) -> dict:

        report_classes = ReportParser._import_all_report_classes()
        result = {}

        for key in keys:
            result[key] = {
                "report": report_classes[key](),
                "params": params[key] if params is not None and key in params else None
            }

        return result
