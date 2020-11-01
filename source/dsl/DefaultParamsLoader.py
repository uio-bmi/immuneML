import logging
import os
import re

import yaml

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.ReflectionHandler import ReflectionHandler


class DefaultParamsLoader:

    @staticmethod
    def _convert_to_snake_case(name):
        if name not in ["MiXCR", "VDJdb"]:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return name.lower()

    @staticmethod
    def _convert_to_camel_case(name):
        return ''.join([i.title() for i in name.split('_')])

    @staticmethod
    def load(path, class_name, log_if_missing=True):
        if os.path.isabs(path):
            filepath = path + DefaultParamsLoader._convert_to_snake_case(class_name) + "_params.yaml"
        else:
            filepath = EnvironmentSettings.default_params_path + path + ("/" if path[-1] != "/" else "") \
                       + DefaultParamsLoader._convert_to_snake_case(class_name) + "_params.yaml"

        if os.path.isfile(filepath):
            with open(filepath, "r") as file:
                params = yaml.load(file, Loader=yaml.FullLoader)
        else:
            if log_if_missing:
                logging.info("DefaultParams: no default parameters were found for {}. Proceeding...".format(class_name))
            params = {}

        return params

    @staticmethod
    def _parse_to_enum_instances(params, location):
        for key in params.keys():
            class_name = DefaultParamsLoader._convert_to_camel_case(key)
            if ReflectionHandler.exists(class_name, location):
                cls = ReflectionHandler.get_class_by_name(class_name, location)
                params[key] = cls[params[key].upper()]
        return params
