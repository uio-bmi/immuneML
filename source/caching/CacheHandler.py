import os
import pickle

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class CacheHandler:

    @staticmethod
    def get(params: tuple):
        h = str(hash(params)).replace("-", "_")
        filename = "{}{}.pickle".format(EnvironmentSettings.cache_path, h)

        obj = None
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                obj = pickle.load(file)

        return obj

    @staticmethod
    def add(params: tuple, caching_object):
        PathBuilder.build(EnvironmentSettings.cache_path)
        h = str(hash(params)).replace("-", "_")
        with open("{}{}.pickle".format(EnvironmentSettings.cache_path, h), "wb") as file:
            pickle.dump(caching_object, file)
