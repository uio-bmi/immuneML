import sys
import os
import argparse
import logging
import warnings
import yaml
import inspect
from pathlib import Path

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.DesignMatrixExporter import DesignMatrixExporter
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReflectionHandler import ReflectionHandler


def get_dummy_dataset(dataset_type):
    if dataset_type == "repertoire":
        return RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=5,
                                                                  sequence_count_probabilities={10: 1},
                                                                  sequence_length_probabilities={10: 1},
                                                                  labels={"my_label": {"yes": 0.5, "no": 0.5},
                                                                          "unnecessary_label": {"yes": 0.5, "no": 0.5}},
                                                                  path=Path("./tmp"))
    elif dataset_type == "receptor":
        return RandomDatasetGenerator.generate_receptor_dataset(receptor_count=5,
                                                                  chain_1_length_probabilities={10: 1},
                                                                  chain_2_length_probabilities={10: 1},
                                                                  labels={"my_label": {"yes": 0.5, "no": 0.5},
                                                                          "unnecessary_label": {"yes": 0.5, "no": 0.5}},
                                                                  path=Path("./tmp"))
    elif dataset_type == "sequence":
        return RandomDatasetGenerator.generate_sequence_dataset(sequence_count=5,
                                                                length_probabilities={10: 1},
                                                                labels={"my_label": {"yes": 0.5, "no": 0.5},
                                                                        "unnecessary_label": {"yes": 0.5, "no": 0.5}},
                                                                path=Path("./tmp"))
def check_default_args(encoder_file_path, no_default_parameters):
    encoder_shortname = encoder_file_path.stem[:-7]
    default_params_filepath = EnvironmentSettings.default_params_path / "encodings" / f"{DefaultParamsLoader.convert_to_snake_case(encoder_shortname)}_params.yaml"


    if no_default_parameters:
        if os.path.isfile(default_params_filepath):
            logging.warning(f"Default parameters file was found at '{default_params_filepath} but 'no_default_parameters' flag was enabled. "
                            f"Assuming no default parameters for further testing, but note that default parameters will be attempted to be loaded by immuneML when using {encoder_file_path.stem}.")
        else:
            logging.info("Skip default parameter testing as 'no_default_parameters' flag is enabled")
        return {}

    logging.info("Checking default parameters file...")

    assert os.path.isfile(default_params_filepath), f"Error: default parameters file was expected to be found at: '{default_params_filepath}'"
    logging.info(f"...Default parameters file was found at: '{default_params_filepath}'")

    logging.info("Attempting to load default parameters file with yaml.load()...")
    with Path(default_params_filepath).open("r") as file:
        default_params = yaml.load(file, Loader=yaml.FullLoader)

    logging.info(f"...The following default parameters were found: '{default_params}'")

    return default_params
    # filepath = EnvironmentSettings.default_params_path / encoder_file_path / f"{DefaultParamsLoader.convert_to_snake_case(encoder_file_path.stem)}_params.yaml"
    # print(filepath)
    # params = DefaultParamsLoader.load(encoder_file_path, encoder_file_path.stem, log_if_missing=True)
    #
    # ObjectParser.get_all_params(specs, class_path, encoder.__name__[:-7], key)
    # print(params)
    pass


def check_methods(encoder_instance):
    logging.info("Checking methods...")
    assert DatasetEncoder.store == encoder_instance.__class__.store, f"Error: class method 'store' should not be overwritten from DatasetEncoder. Found the following: {encoder_instance.__class__.store}"
    assert DatasetEncoder.store_encoder == encoder_instance.__class__.store_encoder, f"Error: class method 'store_encoder' should not be overwritten from DatasetEncoder. Found the following: {encoder_instance.__class__.store_encoder}"
    assert DatasetEncoder.load_attribute == encoder_instance.__class__.load_attribute, f"Error: class method 'load_attribute' should not be overwritten from DatasetEncoder. Found the following: {encoder_instance.__class__.load_attribute}"

    base_class_functions = [name for name, _ in inspect.getmembers(DatasetEncoder, predicate=inspect.isfunction) if not name.startswith("_")]
    class_methods = [name for name, _ in inspect.getmembers(encoder_instance, predicate=inspect.ismethod) if not name.startswith("_")]
    additional_public_methods = [name for name in class_methods if name not in base_class_functions]
    if len(additional_public_methods) > 0:
        logging.warning(f"In immuneML, 'public' methods start without underscore while 'private' methods start with underscore. "
                        f"Methods added to specific encoders other than those inherited from DatasetEncoder should generally be considered private, although this is not strictly enforced. "
                        f"The following additional public methods were found: {', '.join(additional_public_methods)}. To remove this warning, rename these methods to start with _underscore.")

    for method_name in class_methods:
        if method_name != method_name.lower():
            logging.warning(f"Method names must be written in snake_case, found the following method: {method_name}. Please rename this method.")
    logging.info("...Checking methods done.")


def get_dummy_dataset_labels(dummy_dataset):
    if isinstance(dummy_dataset, RepertoireDataset) or isinstance(dummy_dataset, ReceptorDataset):
        return [element.metadata["my_label"] for element in dummy_dataset.get_data()]
    else:
        return [element.get_attribute("my_label") for element in dummy_dataset.get_data()]

def check_labels(enc_data, dummy_dataset):
    labels_descr = "\nLabels must be a dictionary, where: " \
                   "\n- the keys are the label names as found by EncoderParams.label_config.get_labels_by_name()" \
                   "\n- the values are the label values for each example, which must follow the same order as examples in the dataset. Label values can be retrieved using: Repertoire.metadata[label_name], Receptor.metadata[label_name], or Sequence.get_attribute(label_name)."
    assert enc_data.labels is not None, f"Error: EncodedData.labels is None, but labels must be set. " + labels_descr
    assert isinstance(enc_data.labels, dict), f"Error: EncodedData.labels is of type {type(enc_data.labels)}, but must be a dictionary. " + labels_descr

    assert "unnecessary_label" not in enc_data.labels.keys(), "Error: EncodedData.labels contains additional dataset keys which were not specified for encoding. Please make sure only to use the labels given by EncoderParams.label_config.get_labels_by_name()"

    expected = {"my_label": get_dummy_dataset_labels(dummy_dataset)}
    assert set(enc_data.labels.keys()) == {"my_label"}, f"Error: label keys are incorrect. Expected to find the label 'my_dataset', found {enc_data.labels.keys()}. Please make sure only to use the labels given by EncoderParams.label_config.get_labels_by_name()"
    assert enc_data.labels["my_label"] == expected["my_label"], f"Error: values for label 'my_label' are incorrect. Expected  {enc_data.labels['my_label']}, found {expected['my_label']}. Please make sure the label values follow the same order as examples in the dataset. Label values can be retrieved using: Repertoire.metadata[label_name], Receptor.metadata[label_name], or Sequence.get_attribute(label_name)."



def check_encode_method(encoder_instance, dummy_dataset, base_class_name):
    logging.info(f"Testing encode() method functionality...")
    lc = LabelConfiguration()
    lc.add_label(label_name="my_label", values=["yes", "no"], positive_class="yes")
    params = EncoderParams(result_path=Path("./tmp/encoder_results"),
                           label_config=lc)
    encoded_dataset_result = encoder_instance.encode(dummy_dataset, params)

    assert isinstance(encoded_dataset_result, dummy_dataset.__class__), f"Error: expected {encoder_instance.__class__.__name__}.encode() method to return an instance of {dummy_dataset.__class__.__name__} (with attached EncodedData object), found {encoded_dataset_result.__class__.__name__}"
    assert dummy_dataset.get_example_ids() == encoded_dataset_result.get_example_ids(), f"Error: expected the encoded dataset to be a copy of the original dataset (with attached EncodedData object). Found a difference in the example ids: {dummy_dataset.get_example_ids()} != {encoded_dataset_result.get_example_ids()}"

    assert isinstance(encoded_dataset_result.encoded_data, EncodedData), f"Error: expected the .encoded_data field of the output dataset to be an EncodedData object, found {encoded_dataset_result.encoded_data.__class__.__name__}"

    enc_data = encoded_dataset_result.encoded_data

    assert enc_data.examples is not None, f"Error: EncodedData.examples is None, but should be a numeric matrix with a number of rows equal to the number of examples in the dataset ({dummy_dataset.get_example_count()})"
    assert enc_data.examples.shape[0] == dummy_dataset.get_example_count(), f"Error: the number of rows in EncodedData.examples must be equal to the number of examples in the dataset ({dummy_dataset.get_example_count()})"

    assert enc_data.example_ids == dummy_dataset.get_example_ids(), f"Error: EncodedData.example_ids must match the original dataset: {dummy_dataset.get_example_ids()}, found {enc_data.example_ids}"

    check_labels(enc_data, dummy_dataset)

    if enc_data.feature_names is None:
        logging.warning("EncodedData.feature_names is set to None. Please consider adding meaningful feature names to your encoding, as some Reports may crash without feature names.")

    assert enc_data.encoding == base_class_name, f"Error: EncodedData.encoding must be set to the base class name ('{base_class_name}'), found {enc_data.encoding}"

    logging.info("...Testing encode() method functionality done.")
    return encoded_dataset_result

def test_with_design_matrix_exporter(encoded_dataset_result):
    logging.info("Testing exporting the encoded dataset to .csv format with DesignMatrixExporter report...")

    dme_report = DesignMatrixExporter(dataset=encoded_dataset_result, result_path=Path("./tmp/dme_report"), file_format="csv")
    dme_report._generate()

    logging.info("...Testing encoded dataset with DesignMatrixExporter report done.")

def check_encoder_class(encoder_class, dummy_dataset, default_params, base_class_name):
    assert issubclass(encoder_class, DatasetEncoder), f"Error: '{encoder_class.__name__}' is not a subclass of DatasetEncoder"

    logging.info(f"Attempting to run {encoder_class.__name__}.build_object() for datasets of type {dummy_dataset.__class__.__name__} (with default parameters if given)...")
    encoder_instance = encoder_class.build_object(dummy_dataset, **default_params)
    logging.info(f"...Successfully loaded an instance of {encoder_instance.__class__.__name__}")

    check_methods(encoder_instance)
    encoded_dataset_result = check_encode_method(encoder_instance, dummy_dataset, base_class_name)
    test_with_design_matrix_exporter(encoded_dataset_result)


def check_encoder_file(encoder_file):
    assert encoder_file.suffix == ".py", f"Error: '{encoder_file}' is not a Python file"
    assert encoder_file.stem.endswith("Encoder"), f"Error: Filename '{encoder_file}' does not end with 'Encoder'"

    assert encoder_file.parent.parent.stem == "encodings", f"Error: The new DatasetEncoder file(s) must be located inside a new package (folder) under 'immuneML/encodings'. Found the following folder where 'encodings' was expected: {encoder_file.parent.parent}"

def check_args(args):
    assert os.path.isfile(args.encoder_file), f"Error: '{args.encoder_file}' is not a file"



def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for testing new immuneML DatasetEncoder classes")
    parser.add_argument("-e", "--encoder_file", type=str, required=True, help="Path to the (dataset-specific) encoder file, placed in the correct immuneML subfolder. ")
    parser.add_argument("-d", "--dataset_type", type=str, choices=["repertoire", "sequence", "receptor"], required=True, help="Whether to test using 'sequence', 'receptor' or 'repertoire' dataset.")
    parser.add_argument("-p", "--no_default_parameters",  action='store_true', help="If enabled, it is assumed that no default parameters file exists. ")

    return parser.parse_args(args)

def main(args):
    logging.basicConfig(filename="test_new_encoder_log.txt", level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: logging.warning(message)
    tmp_path = Path(os.getcwd()) / "tmp"

    logging.info(f"Storing temporary files at '{tmp_path}'")
    PathBuilder.remove_old_and_build(tmp_path)

    parsed_args = parse_commandline_arguments(args)
    check_args(parsed_args)

    logging.info(f"Testing new encoder file: '{parsed_args.encoder_file}'")

    encoder_file_path = Path(parsed_args.encoder_file)
    check_encoder_file(encoder_file_path)
    encoder_class = ReflectionHandler.get_class_from_path(encoder_file_path, class_name=encoder_file_path.stem)

    default_params = check_default_args(encoder_file_path, parsed_args.no_default_parameters)
    check_encoder_class(encoder_class, get_dummy_dataset(parsed_args.dataset_type), default_params,
                        base_class_name=encoder_file_path.stem)

    logging.info("\n\nCongratulations! If the script ran this far without encountering any error, "
                 f"\nyou have successfully integrated a new DatasetEncoder for {parsed_args.dataset_type} datasets into immuneML. "
                 "\n\nTo finalise your encoder: "
                 "\n- Consider testing the encoder with other dataset types, if applicable."
                 f"\n- Check the exported design matrix to make sure the encoded data fulfills your expectations. This can be found at: {tmp_path/'dme_report/design_matrix.csv'} "
                 "\n- Please take a look if any warnings were thrown in this log file, and consider resolving those. These issues are not critical for the performance of your DatasetEncoder, but may be necessary to resolve before merging the code into the main immuneML branch."
                 "\n- Add class documentation containing a general description of the encoding, its parameters, and an example YAML snippet. "
                 "\n- Add a unit test for the new encoder. "
                 "\n\nFor more details, see https://docs.immuneml.uio.no/latest/developer_docs/how_to_add_new_encoding.html")


if __name__ == "__main__":
    main(sys.argv[1:])