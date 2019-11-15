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
                        print('%s --- Exception in %s : %s' % (datetime.datetime.now(), func.__name__, e))
            finally:
                print("%s --- Exiting: %s" % (datetime.datetime.now(), func.__name__))
        else:
            return func(*args, **kwargs)
    return wrapped


def trace(cls):
    if EnvironmentSettings.log_level != LogLevel.NONE:
        for name, m in inspect.getmembers(cls, lambda x: inspect.isfunction(x) or inspect.ismethod(x)): setattr(cls, name, log(m))
    return cls

