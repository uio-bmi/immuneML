import sys
import os
import yaml
from pathlib import Path
import logging
import warnings
import inspect
from immuneML.util.PathBuilder import PathBuilder

def setup_logger(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file, mode="w"),
                                  logging.StreamHandler(stream=sys.stdout)])
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(message)

def set_tmp_path(tmp_path):
    PathBuilder.remove_old_and_build(tmp_path)
    logging.info(f"Storing temporary files at '{tmp_path}'")

def check_is_alphanum_name(filename_stem):
    for char in filename_stem:
        assert char.isalnum(), f"Error: class name is only allowed to contain alphanumeric characters, found: {char}. Please rename the encoder (both class and filename) to an alphanumeric name."

def load_default_params_file(default_params_filepath):
    logging.info("Checking default parameters file...")

    assert os.path.isfile(default_params_filepath), f"Error: default parameters file was expected to be found at: '{default_params_filepath}'\n" \
                                                    f"To test new classes without default parameters, please run this script with the '-p' flag. "
    logging.info(f"...Default parameters file was found at: '{default_params_filepath}'")

    logging.info("Attempting to load default parameters file with yaml.load()...")
    with Path(default_params_filepath).open("r") as file:
        default_params = yaml.load(file, Loader=yaml.FullLoader)

    logging.info(f"...The following default parameters were found: '{default_params}'")

    return default_params

def check_default_args(default_params_filepath, no_default_parameters, class_name):
    if no_default_parameters:
        if os.path.isfile(default_params_filepath):
            logging.warning(f"Default parameters file was found at '{default_params_filepath} but 'no_default_parameters' flag was enabled. "
                            f"Assuming no default parameters for further testing, but note that default parameters will be attempted to be loaded by immuneML when using {class_name}.")
        else:
            logging.info("Skip default parameter testing as 'no_default_parameters' flag is enabled")
        return {}
    else:
        return load_default_params_file(default_params_filepath)


def check_base_vs_instance_methods(base_class, subclass_instance):
    base_class_functions = [name for name, _ in inspect.getmembers(base_class, predicate=inspect.isfunction) if
                            not name.startswith("_")]
    class_methods = [name for name, _ in inspect.getmembers(subclass_instance, predicate=inspect.ismethod) if
                     not name.startswith("_")]
    additional_public_methods = [name for name in class_methods if name not in base_class_functions]

    if len(additional_public_methods) > 0:
        logging.warning(
            f"In immuneML, 'public' methods start without underscore while 'private' methods start with underscore. "
            f"Methods added to specific encoders other than those inherited from {base_class.__name__} should generally be considered private, although this is not strictly enforced. "
            f"The following additional public methods were found: {', '.join(additional_public_methods)}. To remove this warning, rename these methods to start with _underscore.")

    for method_name in class_methods:
        if method_name != method_name.lower():
            logging.warning(f"Method names must be written in snake_case, found the following method: {method_name}. Please rename this method.")

