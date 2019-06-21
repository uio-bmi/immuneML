import os
import pickle

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class CacheHandler:

    @staticmethod
    def get(params: tuple, cache_type=None):
        h = CacheHandler._hash(params)
        filename = CacheHandler._build_filename(h, cache_type)

        obj = None
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                obj = pickle.load(file)

        return obj

    @staticmethod
    def get_by_key(cache_key: str, cache_type=None):
        filename = CacheHandler._build_filename(cache_key, cache_type)
        obj = None
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                obj = pickle.load(file)
        return obj

    @staticmethod
    def _build_filename(cache_key: str, cache_type=None):
        return "{}{}.pickle".format(EnvironmentSettings.get_cache_path(cache_type), cache_key)

    @staticmethod
    def add(params: tuple, caching_object, cache_type=None):
        PathBuilder.build(EnvironmentSettings.get_cache_path(cache_type))
        h = CacheHandler._hash(params)
        with open(CacheHandler._build_filename(h, cache_type), "wb") as file:
            pickle.dump(caching_object, file)

    @staticmethod
    def add_by_key(cache_key: str, caching_object, cache_type=None):
        PathBuilder.build(EnvironmentSettings.get_cache_path(cache_type))
        with open(CacheHandler._build_filename(cache_key, cache_type), "wb") as file:
            pickle.dump(caching_object, file)

    @staticmethod
    def generate_cache_key(params: tuple, suffix: str):
        return CacheHandler._hash(params) + suffix

    @staticmethod
    def memo(cache_key: str, fn, cache_type=None):
        result = CacheHandler.get_by_key(cache_key, cache_type)
        if result is None:
            result = fn()
            CacheHandler.add_by_key(cache_key, result, cache_type)
        return result

    @staticmethod
    def _hash(params: tuple) -> str:
        return str(hash(params)).replace("-", "_")
