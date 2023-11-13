import datetime
import logging


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
                                    f"see the documentation. For technical description of the error, see the log above.")
                else:
                    raise e
        finally:
            logging.info("--- Exiting: %s" % (func.__name__))

    return wrapped


def print_log(mssg, include_datetime=False, log_func_name='info'):
    getattr(logging, log_func_name)(mssg)

    if include_datetime:
        mssg = f"{datetime.datetime.now()}: {mssg}"

    print(mssg, flush=True)
