import os
import pickle
from unittest import TestCase

from source.caching.CacheHandler import CacheHandler
from source.environment.EnvironmentSettings import EnvironmentSettings


class TestCacheHandler(TestCase):

    def test_get(self):
        params = (("k1", 1), ("k2", 2))
        obj = "object_example"

        h = str(hash(params)).replace("-", "_")
        filename = "{}{}.pickle".format(EnvironmentSettings.cache_path, h)
        with open(filename, "wb") as file:
            pickle.dump(obj, file)

        obj2 = CacheHandler.get(params)
        self.assertEqual(obj, obj2)
        os.remove(filename)

    def test_add(self):
        params = ("k1", 1), ("k2", ("k3", 2))
        obj = "object_example"

        CacheHandler.add(params, obj)

        h = str(hash(params)).replace("-", "_")
        filename = "{}{}.pickle".format(EnvironmentSettings.cache_path, h)
        with open(filename, "rb") as file:
            obj2 = pickle.load(file)

        self.assertEqual(obj, obj2)
        os.remove(filename)
