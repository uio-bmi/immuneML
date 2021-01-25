import hashlib
import logging
import os
import pickle
from pathlib import Path

import dill

from immuneML.caching.CacheObjectType import CacheObjectType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class CacheHandler:

    @staticmethod
    def get_file_path(cache_type=None):
        file_path = EnvironmentSettings.get_cache_path(cache_type) / "files"
        PathBuilder.build(file_path)
        return file_path

    @staticmethod
    def get(params: tuple, object_type, cache_type=None):
        h = CacheHandler._hash(params)
        return CacheHandler.get_by_key(h, object_type, cache_type)

    @staticmethod
    def get_by_key(cache_key: str, object_type, cache_type=None):
        filename = CacheHandler._build_filename(cache_key, object_type, cache_type)
        obj = None
        if filename.is_file():
            with filename.open("rb") as file:
                obj = dill.load(file)
        return obj

    @staticmethod
    def _build_filename(cache_key: str, object_type: CacheObjectType, cache_type=None) -> Path:
        path = EnvironmentSettings.get_cache_path(cache_type) / object_type.name.lower()
        PathBuilder.build(path)
        return path / f"{cache_key}.pickle"

    @staticmethod
    def add(params: tuple, caching_object, object_type: CacheObjectType = CacheObjectType.OTHER, cache_type=None):
        PathBuilder.build(EnvironmentSettings.get_cache_path(cache_type))
        h = CacheHandler.generate_cache_key(params)
        filename = CacheHandler._build_filename(cache_key=h, object_type=object_type, cache_type=cache_type)
        with filename.open("wb") as file:
            dill.dump(caching_object, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def add_by_key(cache_key: str, caching_object, object_type: CacheObjectType = CacheObjectType.OTHER, cache_type=None):
        PathBuilder.build(EnvironmentSettings.get_cache_path(cache_type))
        filename = CacheHandler._build_filename(cache_key=cache_key, object_type=object_type, cache_type=cache_type)
        try:
            with filename.open("wb") as file:
                dill.dump(caching_object, file, protocol=pickle.HIGHEST_PROTOCOL)
        except AttributeError:
            os.remove(filename)
            logging.warning(f"CacheHandler: could not cache object of class {type(caching_object).__name__} with key {cache_key}. "
                            f"Object: {caching_object}\n"
                            f"Next time this object is needed, it will be recomputed which will take more time but should not influence results.")

    @staticmethod
    def generate_cache_key(params: tuple):
        return hashlib.sha256(str(params).encode('utf-8')).hexdigest()

    @staticmethod
    def memo(cache_key: str, fn, object_type: CacheObjectType = CacheObjectType.OTHER, cache_type=None):
        result = CacheHandler.get_by_key(cache_key, object_type, cache_type)
        if result is None:
            result = fn()
            CacheHandler.add_by_key(cache_key, result, object_type, cache_type)
        return result

    @staticmethod
    def memo_by_params(params: tuple, fn, object_type: CacheObjectType = CacheObjectType.OTHER, cache_type=None):
        cache_key = CacheHandler.generate_cache_key(params)
        return CacheHandler.memo(cache_key, fn, object_type, cache_type)

    @staticmethod
    def _hash(params: tuple) -> str:
        return hashlib.sha256(str(params).encode('utf-8')).hexdigest()
