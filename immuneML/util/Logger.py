import datetime
import logging

import psutil


def log(func):
    def wrapped(*args, **kwargs):
        try:
            logging.info("--- Entering: %s with parameters %s" % (func.__name__, args))
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error('\n\n--- Exception in %s : %s\n\n' % (func.__name__, e))
                if "dsl" in func.__globals__["__name__"]:
                    raise Exception(f"{e}\n\n"
                                    f"ImmuneMLParser: an error occurred during parsing in function {func.__name__} "
                                    f" with parameters: {args}.\n\nFor more details on how to write the specification, "
                                    f"see the documentation. For technical description of the error, see the log above.").with_traceback(e.__traceback__)
                else:
                    raise e
        finally:
            logging.info("--- Exiting: %s" % (func.__name__))

    return wrapped


def print_log(mssg, include_datetime=True, log_func_name='info'):
    getattr(logging, log_func_name)(mssg)

    if include_datetime:
        mssg = f"{datetime.datetime.now()}: {mssg}"

    print(mssg, flush=True)


def log_memory_usage(stage: str, location: str = None):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    logging.info(f"{location}: Memory usage at {stage}: {memory_gb:.2f} GB")
