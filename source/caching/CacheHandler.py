import os
import pickle

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class CacheHandler:

    @staticmethod
    def get(params: tuple):
        h = CacheHandler._hash(params)
        filename = CacheHandler._build_filename(h)

        obj = None
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                obj = pickle.load(file)

        return obj

    @staticmethod
    def get_by_key(cache_key: str):
        filename = CacheHandler._build_filename(cache_key)
        obj = None
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                obj = pickle.load(file)
        return obj

    @staticmethod
    def _build_filename(cache_key: str):
        return "{}{}.pickle".format(EnvironmentSettings.cache_path, cache_key)

    @staticmethod
    def add(params: tuple, caching_object):
        PathBuilder.build(EnvironmentSettings.cache_path)
        h = CacheHandler._hash(params)
        with open(CacheHandler._build_filename(h), "wb") as file:
            pickle.dump(caching_object, file)

    @staticmethod
    def add_by_key(cache_key: str, caching_object):
        PathBuilder.build(EnvironmentSettings.cache_path)
        with open(CacheHandler._build_filename(cache_key), "wb") as file:
            pickle.dump(caching_object, file)

    @staticmethod
    def generate_cache_key(params: tuple, suffix: str):
        return CacheHandler._hash(params) + suffix

    @staticmethod
    def memo(cache_key: str, fn):
        result = CacheHandler.get_by_key(cache_key)
        if result is None:
            result = fn()
            CacheHandler.add_by_key(cache_key, result)
        return result

    @staticmethod
    def _hash(params: tuple) -> str:
        return str(hash(params)).replace("-", "_")
