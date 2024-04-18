import argparse
import random

import numpy as np

from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.checker_util import *


def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for testing new immuneML MLMethod classes")
    parser.add_argument("-m", "--ml_method_file", type=str, required=True, help="Path to the MLMethod file, placed in the correct immuneML subfolder. ")
    parser.add_argument("-p", "--no_default_parameters",  action='store_true', help="If enabled, it is assumed that no default parameters file exists, and the MLMethod can be run without supplying additional parameters. ")
    parser.add_argument("-l", "--log_file", type=str, default="check_new_ml_method_log.txt", help="Path to the output log file. If already present, the file will be overwritten.")
    parser.add_argument("-t", "--tmp_path", type=str, default="./tmp", help="Path to the temporary output folder. If already present, the folder will be overwritten.")

    return parser.parse_args(args)


def check_ml_method_file(ml_method_file):
    logging.info(f"Checking file name and location...")

    assert ml_method_file.parent.stem == "classifiers", f"Error: The new MLMethod file must be located inside a new package (folder) under 'immuneML/ml_methods/classifiers'. Found the following folder where 'classifiers' was expected: {ml_method_file.parent}"
    assert ml_method_file.parent.parent.stem == "ml_methods", f"Error: The new MLMethod file must be located inside a new package (folder) under 'immuneML/ml_methods/classifiers'. Found the following folder where 'ml_methods' was expected: {ml_method_file.parent}"

    assert ml_method_file.suffix == ".py", f"Error: '{ml_method_file}' is not a Python file"
    assert ml_method_file.stem[0].isupper(), f"Error: Filename '{ml_method_file}' does not start with an upper case character. Please rename the ML method (both class and filename) to '{ml_method_file.stem.title()}'"

    check_is_alphanum_name(ml_method_file.stem)

    logging.info(f"...Checking file name and location done.")

def get_ml_method_class(ml_method_file_path):
    logging.info(f"Attempting to load MLMethod class named '{ml_method_file_path.stem}'...")
    ml_method_class = ReflectionHandler.get_class_from_path(ml_method_file_path, class_name=ml_method_file_path.stem)
    logging.info(f"...Loading MLMethod class done.")

    return ml_method_class

def get_default_params_filepath_from_name(ml_method_file_path):
    return EnvironmentSettings.default_params_path / "ml_methods" / f"{DefaultParamsLoader.convert_to_snake_case(ml_method_file_path.stem)}_params.yaml"

def check_methods(ml_method_instance):
    logging.info("Checking methods...")
    mssg = "Error: class method '{}' should not be overwritten from MLMethod. Found the following: {}"

    assert MLMethod.fit == ml_method_instance.__class__.fit, mssg.format("fit", ml_method_instance.__class__.fit)
    assert MLMethod._initialize_fit == ml_method_instance.__class__._initialize_fit, mssg.format("_initialize_fit", ml_method_instance.__class__._initialize_fit)
    assert MLMethod.fit_by_cross_validation == ml_method_instance.__class__.fit_by_cross_validation, mssg.format("fit_by_cross_validation", ml_method_instance.__class__.fit_by_cross_validation)
    assert MLMethod._assert_matching_label == ml_method_instance.__class__._assert_matching_label, mssg.format("_assert_matching_label", ml_method_instance.__class__._assert_matching_label)
    assert MLMethod.predict == ml_method_instance.__class__.predict, mssg.format("predict", ml_method_instance.__class__.predict)
    assert MLMethod.predict_proba == ml_method_instance.__class__.predict_proba, mssg.format("predict_proba", ml_method_instance.__class__.predict_proba)
    assert MLMethod.check_encoder_compatibility == ml_method_instance.__class__.check_encoder_compatibility, mssg.format("check_encoder_compatibility", ml_method_instance.__class__.check_encoder_compatibility)
    assert MLMethod.get_feature_names == ml_method_instance.__class__.get_feature_names, mssg.format("get_feature_names", ml_method_instance.__class__.get_feature_names)
    assert MLMethod.get_label_name == ml_method_instance.__class__.get_label_name, mssg.format("get_label_name", ml_method_instance.__class__.get_label_name)
    assert MLMethod.get_classes == ml_method_instance.__class__.get_classes, mssg.format("get_classes", ml_method_instance.__class__.get_classes)
    assert MLMethod.get_positive_class == ml_method_instance.__class__.get_positive_class, mssg.format("get_positive_class", ml_method_instance.__class__.get_positive_class)

    check_base_vs_instance_methods(MLMethod, ml_method_instance)

    compatible_encoders = ml_method_instance.get_compatible_encoders()
    assert isinstance(compatible_encoders, list), f"get_compatible_encoders() should return a list of compatible encoder classes (found: {type(compatible_encoders)}). If no compatible encoder exists, it should return an empty list. "
    if ml_method_instance.get_compatible_encoders() == []:
        logging.warning(f"get_compatible_encoders() returned an empty list. At least one encoder class must be returned in this list in order to be able to use the MLMethod in practice.")

    assert isinstance(ml_method_instance.can_predict_proba(), bool), f"can_predict_proba() should return boolean value (found: {type(ml_method_instance.can_predict_proba())}). "
    assert isinstance(ml_method_instance.can_fit_with_example_weights(), bool), f"can_fit_with_example_weights() should return boolean value (found: {type(ml_method_instance.can_fit_with_example_weights())}). "

    logging.info("...Checking methods done.")

def get_example_encoded_data():
    """
    Note: by default, this random EncodedData object is used to represent an encoded dataset.
    Some MLMethods require a more specific encoding type and will fail with this example
    for instance: if different dimensions or additional info are used in the encoded data.

    You may overwrite this method to supply a custom EncodedData object which meets the requirements of your MLMethod.
    """
    enc_data = EncodedData(examples=np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 1, 1]]),
                           example_ids=list(range(8)),
                           feature_names=["a", "b", "c"],
                           labels={"my_label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]},
                           encoding="random")

    label = Label(name="my_label", values=["yes", "no"], positive_class="yes")

    return enc_data, label

def check_params_recursively(params):
    if params is None or isinstance(params, (str, float, int)):
        return
    elif isinstance(params, list):
        for param in params:
            check_params_recursively(param)
    elif isinstance(params, dict):
        for key, value in params.items():
            check_params_recursively(key)
            check_params_recursively(value)
    else:
        raise ValueError(f"get_params() should return a yaml-friendly parameter representation, consisting only of lists, dicts and strings. Found: {params} of type {type(params)}")

def check_predictions(ml_method_instance, predictions, label, enc_data):
    error_mssg = f"The output of {ml_method_instance.__class__.__name__}.predict() is assumed to be a dictionary where the label name is the key, and its value is a numpy array with classes (one prediction per example)." \
                 "\nFor example: {" + f"'{label.name}': {[random.choice(label.values) for i in range(enc_data.examples.shape[0])]}" + "}" \
                                                                                                                                      f"\nFound the follwing instead: {predictions}"
    assert isinstance(predictions, dict), error_mssg + f"\n\nFound the wrong type: {type(predictions)}"
    assert list(predictions.keys()) == [label.name], error_mssg + f"\n\nFound the wrong keys: {list(predictions.keys())}"

    assert len(list(predictions["my_label"])) == enc_data.examples.shape[0], error_mssg + f"\n\nFound the wrong number of predictions."
    for pred in predictions["my_label"]:
        assert pred in label.values, error_mssg + f"\n\nFound the wrong predicted class: {pred}"

def check_predictions_proba(ml_method_instance, proba_predictions, label, enc_data):
    class1_proba = [round(random.uniform(0, 1), 2) for p in range(enc_data.examples.shape[0])]
    class2_proba = [round(1 - p, 2) for p in class1_proba]

    error_mssg = f"The output of {ml_method_instance.__class__.__name__}.predict_proba() is assumed to be a dictionary where the label name is the key," \
                 f"and its value is a dictionary containing the probabilities for each class." \
                 "\nFor example (numbers are random examples): {" + f"'{label.name}': " + "{" + f"'{label.values[1]}': {class1_proba}, '{label.values[0]}': {class2_proba}" + "}}" \
                f"\nFound the follwing instead: {proba_predictions}"

    assert isinstance(proba_predictions, dict), error_mssg + f"\n\nFound the wrong type: {type(proba_predictions)}"
    assert list(proba_predictions.keys()) == [label.name], error_mssg + f"\n\nFound the wrong keys: {list(proba_predictions.keys())}"

    assert isinstance(proba_predictions[label.name], dict), error_mssg + f"\n\nFound the wrong type where an inner dict was found: {type(proba_predictions[label.name])}"
    assert sorted(proba_predictions[label.name].keys()) == sorted(label.values), error_mssg + f"\n\nFound the wrong keys for the inner dict: {proba_predictions[label.name].keys()}"

    proba1 = proba_predictions[label.name][label.values[0]]
    proba2 = proba_predictions[label.name][label.values[1]]

    assert len(proba1) == enc_data.examples.shape[0], error_mssg + f"\n\nFound the wrong number of predictions: {proba1} (expected to be of length {enc_data.shape[0]})"
    assert len(proba2) == enc_data.examples.shape[0], error_mssg + f"\n\nFound the wrong number of predictions: {proba2} (expected to be of length {enc_data.shape[0]})"

    for pred1, pred2 in zip(proba1, proba2):
        assert isinstance(pred1, float), error_mssg + f"\n\nFound the following value of type {type(pred1)} where float was expected: {pred1}"
        assert isinstance(pred2, float), error_mssg + f"\n\nFound the following value of type {type(pred2)} where float was expected: {pred2}"

        assert 1 >= pred1 >= 0, error_mssg + f"\n\nExpected all predictions to range between [0, 1], found: {pred1}"
        assert 1 >= pred2 >= 0, error_mssg + f"\n\nExpected all predictions to range between [0, 1], found: {pred2}"

        assert 1.01 >= pred1+pred2 >= 0.99, error_mssg + f"\n\nProbabilties for both classes are expected to (approximately) sum to 1, found: {pred1} + {pred2} = {pred1 + pred2}"

def check_model_fitting_and_prediction(ml_method_instance, tmp_path):
    enc_data, label = get_example_encoded_data()

    logging.info(f"Attempting to call {ml_method_instance.__class__.__name__}.fit()...")
    ml_method_instance.fit(enc_data, label, optimization_metric="balanced_accuracy")
    logging.info(f"...Succeeded to calling {ml_method_instance.__class__.__name__}.fit()...")

    logging.info(f"Attempting to call {ml_method_instance.__class__.__name__}.predict()...")
    predictions = ml_method_instance.predict(enc_data, label)
    check_predictions(ml_method_instance, predictions, label, enc_data)
    logging.info(f"...Succeeded calling {ml_method_instance.__class__.__name__}.predict(). Predictions are:\n{predictions}")

    if ml_method_instance.can_predict_proba():
        logging.info("Attempting to predict ")
        logging.info(f"{ml_method_instance.__class__.__name__}.can_predict_proba() returns True, attempting to call {ml_method_instance.__class__.__name__}.predict_proba()...")
        proba_predictions = ml_method_instance.predict_proba(enc_data, label)
        check_predictions_proba(ml_method_instance, proba_predictions, label, enc_data)
        logging.info(f"...Succeeded calling {ml_method_instance.__class__.__name__}.predict_proba(). Predictions are:\n{proba_predictions}")
    else:
        logging.info(f"{ml_method_instance.__class__.__name__}.can_predict_proba() returns False, {ml_method_instance.__class__.__name__}.predict_proba() will be ignored")

    logging.info(f"Checking {ml_method_instance.__class__.__name__}.get_params()...")
    params = ml_method_instance.get_params()
    logging.info(f"Checking formatting of the following params: {params}")
    check_params_recursively(params)

    with (tmp_path / "get_params.yaml").open("w") as file:
        yaml.dump(params, file)
    logging.info(f"...checking {ml_method_instance.__class__.__name__}.get_params() done.")

def check_store_load(ml_method_instance, tmp_path):
    logging.info(f"Attempting to store and load {ml_method_instance.__class__.__name__}...")

    logging.info(f"Storing to {tmp_path}...")
    ml_method_instance.store(tmp_path)
    logging.info(f"...Storing done")
    logging.info(f"Loading model from {tmp_path}...")

    loaded_model = ReflectionHandler.get_class_by_name(ml_method_instance.__class__.__name__, 'ml_methods/')()
    loaded_model.load(tmp_path)
    logging.info(f"...Loading done")

    logging.info(f"Testing if original and re-loaded model make the same predictions...")
    example_data, label = get_example_encoded_data()
    orig_preds = ml_method_instance.predict(example_data, label)
    loaded_preds = loaded_model.predict(example_data, label)

    assert (orig_preds["my_label"] == loaded_preds["my_label"]).all(), f"When using the same example data to call .predict() on both the original model " \
                                                                       f"and the model loaded after storing, different results were found: " \
                                                                       f"\nOriginal model predictions: {orig_preds}" \
                                                                       f"\nLoaded model predictions: {orig_preds}." \
                                                                       f"\nThis indicates an issue with storing and loading the model."

    logging.info(f"...The original and re-loaded model make the same predictions.")

    logging.info(f"...Succeeded storing and loading.")

def check_ml_method_class(ml_method_class, default_params, tmp_path):
    assert issubclass(ml_method_class, MLMethod), f"Error: '{ml_method_class.__name__}' is not a subclass of MLMethod"

    logging.info(f"Attempting to create a new {ml_method_class.__name__} object (with default parameters if given)...")
    ml_method_instance = ml_method_class(**default_params)
    logging.info(f"...Successfully created an instance of {ml_method_instance.__class__.__name__}")

    check_methods(ml_method_instance)
    check_model_fitting_and_prediction(ml_method_instance, tmp_path)
    check_store_load(ml_method_instance, tmp_path)

def run_checks(parsed_args):
    logging.info(f"Testing new ML method file: '{parsed_args.ml_method_file}'")

    assert os.path.isfile(parsed_args.ml_method_file), f"Error: '{parsed_args.ml_method_file}' is not a file"
    ml_method_file_path = Path(parsed_args.ml_method_file)

    check_ml_method_file(ml_method_file_path)
    ml_method_class = get_ml_method_class(ml_method_file_path)

    default_params_filepath = get_default_params_filepath_from_name(ml_method_file_path)
    default_params = check_default_args(default_params_filepath, parsed_args.no_default_parameters, class_name=ml_method_file_path.stem)

    check_ml_method_class(ml_method_class, default_params, tmp_path=Path(parsed_args.tmp_path))

    logging.info("\n\nCongratulations! If the script ran this far without encountering any error, "
                 f"\nyou have successfully integrated a new MLMethod into immuneML. "
                 "\n\nTo finalise your ML method: "
                 "\n- Test run your MLMethod with an appropriate Encoder using the TrainMLModel instruction. See examples in the immuneML documentation at https://docs.immuneml.uio.no"
                 "\n- Please take a look if any warnings were thrown in this log file, and consider resolving those. These issues are not critical for the performance of your MLMethod, but may be necessary to resolve before merging the code into the main immuneML branch."
                 "\n- Add class documentation containing a general description of the encoding, its parameters, and an example YAML snippet. "
                 "\n- Add a unit test for the new ML method. "
                 "\n\nFor more details, see https://docs.immuneml.uio.no/latest/developer_docs/how_to_add_new_ML_method.html")

def main(args):
    parsed_args = parse_commandline_arguments(args)

    setup_logger(parsed_args.log_file)
    set_tmp_path(parsed_args.tmp_path)

    try:
        run_checks(parsed_args)
    except Exception as e:
        logging.error("\n\nA critical error occurred when testing the new MLMethod:\n")
        logging.exception(e)

        raise e

if __name__ == "__main__":
    main(sys.argv[1:])
