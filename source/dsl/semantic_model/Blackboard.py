import warnings


class Blackboard:

    def __init__(self, objects: dict = None):
        self._objects = objects if objects is not None else {}

    def add(self, key: str, value):
        if key in self._objects.keys():
            warnings.warn("Overwriting existing object in the blackboard with key {}...".format(key))

        self._objects[key] = value

    def get(self, key: str):
        if key in self._objects.keys():
            return self._objects[key]
        else:
            return None
