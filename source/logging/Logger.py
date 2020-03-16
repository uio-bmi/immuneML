import datetime
import inspect

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.logging.LogLevel import LogLevel


def log(func):
    def wrapped(*args, **kwargs):
        if EnvironmentSettings.log_level != LogLevel.NONE:
            try:
                print("%s --- Entering: %s with parameters %s" % (datetime.datetime.now(), func.__name__, args))
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if EnvironmentSettings.log_level == LogLevel.DEBUG:
                        print('\n\n%s --- Exception in %s : %s\n\n' % (datetime.datetime.now(), func.__name__, e))
                    if "dsl" in func.__globals__["__name__"]:
                        raise Exception(f"{e}\n\n"
                                        f"ImmuneMLParser: an error occurred during parsing in function {func.__name__} "
                                        f" with parameters: {args}.\n\nFor more details on how to write the specification, "
                                        f"see the documentation. For technical description of the error, see the log above.")
                    else:
                        raise e
            finally:
                print("%s --- Exiting: %s" % (datetime.datetime.now(), func.__name__))
        else:
            return func(*args, **kwargs)
    return wrapped


def trace(cls):
    if EnvironmentSettings.log_level != LogLevel.NONE:
        for name, m in inspect.getmembers(cls, lambda x: inspect.isfunction(x) or inspect.ismethod(x)): setattr(cls, name, log(m))
    return cls

