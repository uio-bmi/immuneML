from importlib import import_module


class MLParser:

    @staticmethod
    def parse_ml_methods(ml_methods) -> list:

        if isinstance(ml_methods, list):
            methods = MLParser._build_default_methods(ml_methods)
        else:
            methods = MLParser._build_methods_with_params(ml_methods)

        return methods

    @staticmethod
    def _build_default_methods(ml_methods: list) -> list:
        methods = []

        for key in ml_methods:
            mod = import_module("source.ml_methods.{}".format(key))
            method = getattr(mod, key)()
            methods.append(method)

        return methods

    @staticmethod
    def _build_methods_with_params(ml_methods: dict) -> list:
        methods = []
        for key in ml_methods.keys():

            param_grid = {k: ml_methods[key][k] if isinstance(ml_methods[key][k], list) else [ml_methods[key][k]]
                          for k in ml_methods[key].keys()}

            mod = import_module("source.ml_methods.{}".format(key))
            method = getattr(mod, key)(parameter_grid=param_grid)
            methods.append(method)

        return methods
