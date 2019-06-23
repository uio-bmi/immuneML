import hashlib
import os
import pickle
from unittest import TestCase

from source.caching.CacheHandler import CacheHandler
from source.environment.EnvironmentSettings import EnvironmentSettings


class TestCacheHandler(TestCase):

    def test_get(self):
        params = (("k1", 1), ("k2", 2))
        obj = "object_example"

        h = hashlib.sha256(str(params).encode('utf-8')).hexdigest()
        filename = "{}{}.pickle".format(EnvironmentSettings.get_cache_path(), h)
        with open(filename, "wb") as file:
            pickle.dump(obj, file)

        obj2 = CacheHandler.get(params)
        self.assertEqual(obj, obj2)
        os.remove(filename)

    def test_get_by_key(self):
        params = (("k1", 1), ("k2", 2))
        obj = "object_example"

        h = CacheHandler._hash(params)
        filename = CacheHandler._build_filename(h)
        with open(filename, "wb") as file:
            pickle.dump(obj, file)

        obj2 = CacheHandler.get_by_key(h)
        self.assertEqual(obj, obj2)
        os.remove(filename)

    def test_add(self):
        params = ("k1", 1), ("k2", ("k3", 2))
        obj = "object_example"

        CacheHandler.add(params, obj)

        h = CacheHandler._hash(params)
        filename = CacheHandler._build_filename(h)
        with open(filename, "rb") as file:
            obj2 = pickle.load(file)

        self.assertEqual(obj, obj2)
        os.remove(filename)

    def test_add_by_key(self):
        params = ("k1", 1), ("k2", ("k3", 2))
        obj = "object_example"

        h = CacheHandler._hash(params)
        CacheHandler.add_by_key(h, obj)
        filename = CacheHandler._build_filename(h)
        with open(filename, "rb") as file:
            obj2 = pickle.load(file)

        self.assertEqual(obj, obj2)
        os.remove(filename)

    def test_memo(self):
        fn = lambda: "abc"
        cache_key = "a123"
        obj = CacheHandler.memo(cache_key, fn)
        self.assertEqual("abc", obj)

        os.remove(CacheHandler._build_filename(cache_key))
