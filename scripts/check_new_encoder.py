import argparse

from scripts.checker_util import *
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
from immuneML.util.ReflectionHandler import ReflectionHandler


def parse_commandline_arguments(args):
    parser = argparse.ArgumentParser(description="Tool for testing new immuneML DatasetEncoder classes")
    parser.add_argument("-e", "--encoder_file", type=str, required=True, help="Path to the (dataset-specific) encoder file, placed in the correct immuneML subfolder. ")
    parser.add_argument("-d", "--dataset_type", type=str, choices=["repertoire", "sequence", "receptor"], required=True, help="Whether to test using 'sequence', 'receptor' or 'repertoire' dataset.")
    parser.add_argument("-p", "--no_default_parameters",  action='store_true', help="If enabled, it is assumed that no default parameters file exists, and the Encoder can be run without supplying additional parameters. ")
    parser.add_argument("-l", "--log_file", type=str, default="check_new_encoder_log.txt", help="Path to the output log file. If already present, the file will be overwritten.")
    parser.add_argument("-t", "--tmp_path", type=str, default="./tmp", help="Path to the temporary output folder. If already present, the folder will be overwritten.")

    return parser.parse_args(args)

def check_methods(encoder_instance):
    logging.info("Checking methods...")
    mssg = "Error: class method '{}' should not be overwritten from DatasetEncoder. Found the following: {}"

    assert DatasetEncoder.store == encoder_instance.__class__.store, mssg.format("store", encoder_instance.__class__.store)
    assert DatasetEncoder.store_encoder == encoder_instance.__class__.store_encoder, mssg.format("store_encoder", encoder_instance.__class__.store_encoder)
    assert DatasetEncoder.load_attribute == encoder_instance.__class__.load_attribute, mssg.format("load_attribute", encoder_instance.__class__.load_attribute)

    check_base_vs_instance_methods(DatasetEncoder, encoder_instance)

    logging.info("...Checking methods done.")

def check_encoded_dataset(encoded_dataset_result, dummy_dataset, encoder_instance):
    assert isinstance(encoded_dataset_result, dummy_dataset.__class__), f"Error: expected {encoder_instance.__class__.__name__}.encode() method to return an instance of {dummy_dataset.__class__.__name__} (with attached EncodedData object), found {encoded_dataset_result.__class__.__name__}"
    assert dummy_dataset.get_example_ids() == encoded_dataset_result.get_example_ids(), f"Error: expected the encoded dataset to be a copy of the original dataset (with attached EncodedData object). Found a difference in the example ids: {dummy_dataset.get_example_ids()} != {encoded_dataset_result.get_example_ids()}"

def check_encoded_data(encoded_data, dummy_dataset, base_class_name):
    assert isinstance(encoded_data, EncodedData), f"Error: expected the .encoded_data field of the output dataset to be an EncodedData object, found {encoded_data.__class__.__name__}"

    assert encoded_data.examples is not None, f"Error: EncodedData.examples is None, but should be a numeric matrix with a number of rows equal to the number of examples in the dataset ({dummy_dataset.get_example_count()})"
    assert encoded_data.examples.shape[0] == dummy_dataset.get_example_count(), f"Error: the number of rows in EncodedData.examples must be equal to the number of examples in the dataset ({dummy_dataset.get_example_count()})"

    assert encoded_data.example_ids == dummy_dataset.get_example_ids(), f"Error: EncodedData.example_ids must match the original dataset: {dummy_dataset.get_example_ids()}, found {encoded_data.example_ids}"
    assert encoded_data.encoding == base_class_name, f"Error: EncodedData.encoding must be set to the base class name ('{base_class_name}'), found {encoded_data.encoding}"

    if encoded_data.feature_names is None:
        logging.warning("EncodedData.feature_names is set to None. Please consider adding meaningful feature names to your encoding, as some Reports may crash without feature names.")

def get_dummy_dataset_labels(dummy_dataset):
    if isinstance(dummy_dataset, RepertoireDataset) or isinstance(dummy_dataset, ReceptorDataset):
        return [element.metadata["my_label"] for element in dummy_dataset.get_data()]
    elif isinstance(dummy_dataset, SequenceDataset):
        return [element.get_attribute("my_label") for element in dummy_dataset.get_data()]

def check_labels(encoded_data, dummy_dataset):
    labels_descr = "\nLabels must be a dictionary, where: " \
                   "\n- the keys are the label names as found by EncoderParams.label_config.get_labels_by_name()" \
                   "\n- the values are the label values for each example, which must follow the same order as examples in the dataset. Label values can be retrieved using: Repertoire.metadata[label_name], Receptor.metadata[label_name], or Sequence.get_attribute(label_name)."
    assert encoded_data.labels is not None, f"Error: EncodedData.labels is None, but labels must be set. " + labels_descr
    assert isinstance(encoded_data.labels, dict), f"Error: EncodedData.labels is of type {type(encoded_data.labels)}, but must be a dictionary. " + labels_descr

    assert "unnecessary_label" not in encoded_data.labels.keys(), "Error: EncodedData.labels contains additional dataset keys which were not specified for encoding. Please make sure only to use the labels given by EncoderParams.label_config.get_labels_by_name()"

    expected = {"my_label": get_dummy_dataset_labels(dummy_dataset)}
    assert set(encoded_data.labels.keys()) == {"my_label"}, f"Error: label keys are incorrect. Expected to find the label 'my_dataset', found {encoded_data.labels.keys()}. Please make sure only to use the labels given by EncoderParams.label_config.get_labels_by_name()"
    assert encoded_data.labels["my_label"] == expected["my_label"], f"Error: values for label 'my_label' are incorrect. Expected  {encoded_data.labels['my_label']}, found {expected['my_label']}. Please make sure the label values follow the same order as examples in the dataset. Label values can be retrieved using: Repertoire.metadata[label_name], Receptor.metadata[label_name], or Sequence.get_attribute(label_name)."

def check_encode_method_with_labels(encoder_instance, dummy_dataset, base_class_name, tmp_path):
    logging.info(f"Testing encode() method functionality with labels...")

    lc = LabelConfiguration()
    lc.add_label(label_name="my_label", values=["yes", "no"], positive_class="yes")
    params = EncoderParams(result_path=Path(tmp_path) / "encoder_results",
                           label_config=lc,
                           encode_labels=True)
    encoded_dataset_result = encoder_instance.encode(dummy_dataset, params)

    check_encoded_dataset(encoded_dataset_result, dummy_dataset, encoder_instance)
    check_encoded_data(encoded_dataset_result.encoded_data, dummy_dataset, base_class_name)
    check_labels(encoded_dataset_result.encoded_data, dummy_dataset)

    logging.info("...Testing encode() method functionality with labels done.")

    return encoded_dataset_result

def check_encode_method_without_labels(encoder_instance, dummy_dataset, base_class_name, tmp_path):
    logging.info(f"Testing encode() method functionality without labels...")

    params = EncoderParams(result_path=Path(tmp_path) / "encoder_results_no_labels",
                           label_config=LabelConfiguration(),
                           encode_labels=False)
    encoded_dataset_result = encoder_instance.encode(dummy_dataset, params)

    check_encoded_dataset(encoded_dataset_result, dummy_dataset, encoder_instance)
    check_encoded_data(encoded_dataset_result.encoded_data, dummy_dataset, base_class_name)

    assert encoded_dataset_result.encoded_data.labels is None, f"Error: EncodedData.labels must be set to 'None' if EncoderParams.encode_labels is False. Found the following instead: {encoded_dataset_result.encoded_data.labels}"
    logging.info("...Testing encode() method functionality without labels done.")

    return encoded_dataset_result

def check_encode_method(encoder_instance, dummy_dataset, base_class_name, tmp_path):
    logging.info(f"Testing encode() method functionality...")

    encode_method_result_no_labels = check_encode_method_without_labels(encoder_instance, dummy_dataset, base_class_name, tmp_path)
    encode_method_result = check_encode_method_with_labels(encoder_instance, dummy_dataset, base_class_name, tmp_path)

    logging.info("...Testing encode() method functionality done.")

    return encode_method_result

def test_with_design_matrix_exporter(encoded_dataset_result, tmp_path):
    logging.info("Testing exporting the encoded dataset to .csv format with DesignMatrixExporter report...")

    dme_report = DesignMatrixExporter(dataset=encoded_dataset_result, result_path=Path(tmp_path) / "dme_report", file_format="csv")
    dme_report._generate()

    logging.info("...Testing exporting the encoded dataset to .csv format with DesignMatrixExporter report done.")

def check_encoder_class(encoder_class, dummy_dataset, default_params, base_class_name, tmp_path):
    assert issubclass(encoder_class, DatasetEncoder), f"Error: '{encoder_class.__name__}' is not a subclass of DatasetEncoder"

    logging.info(f"Attempting to run {encoder_class.__name__}.build_object() for datasets of type {dummy_dataset.__class__.__name__} (with default parameters if given)...")
    encoder_instance = encoder_class.build_object(dummy_dataset, **default_params)
    logging.info(f"...Successfully loaded an instance of {encoder_instance.__class__.__name__}")

    check_methods(encoder_instance)
    encoded_dataset_result = check_encode_method(encoder_instance, dummy_dataset, base_class_name, tmp_path)
    test_with_design_matrix_exporter(encoded_dataset_result, tmp_path=tmp_path)


def check_encoder_file(encoder_file):
    logging.info(f"Checking file name and location...")
    assert encoder_file.parent.parent.stem == "encodings", f"Error: The new DatasetEncoder file(s) must be located inside a new package (folder) under 'immuneML/encodings'. Found the following folder where 'encodings' was expected: {encoder_file.parent.parent}"

    assert encoder_file.suffix == ".py", f"Error: '{encoder_file}' is not a Python file"
    assert encoder_file.stem.endswith("Encoder"), f"Error: Filename '{encoder_file}' does not end with 'Encoder'"
    assert encoder_file.stem[0].isupper(), f"Error: Filename '{encoder_file}' does not start with an upper case character. Please rename the encoder (both class and filename) to '{encoder_file.stem.title()}'"

    check_is_alphanum_name(encoder_file.stem)

    init_file_path = encoder_file.parent / "__init__.py"
    if not init_file_path.is_file():
        with open(init_file_path, 'w') as file:
            pass
        logging.info(f"Expected an empty init file to be found at {init_file_path}. This file was not present previously, but has now been created.")

    logging.info(f"...Checking file name and location done.")

def get_encoder_class(encoder_file_path):
    logging.info(f"Attempting to load Encoder class named '{encoder_file_path.stem}'...")
    encoder_class = ReflectionHandler.get_class_from_path(encoder_file_path, class_name=encoder_file_path.stem)
    logging.info(f"...Loading Encoder class done.")

    return encoder_class

def get_default_params_filepath_from_name(encoder_file_path):
    encoder_shortname = encoder_file_path.stem[:-7]
    return EnvironmentSettings.default_params_path / "encodings" / f"{DefaultParamsLoader.convert_to_snake_case(encoder_shortname)}_params.yaml"


def get_dummy_dataset(dataset_type, tmp_path):
    if dataset_type == "repertoire":
        return RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=5,
                                                                  sequence_count_probabilities={10: 1},
                                                                  sequence_length_probabilities={10: 1},
                                                                  labels={"my_label": {"yes": 0.5, "no": 0.5},
                                                                          "unnecessary_label": {"yes": 0.5, "no": 0.5}},
                                                                  path=Path(tmp_path))
    elif dataset_type == "receptor":
        return RandomDatasetGenerator.generate_receptor_dataset(receptor_count=5,
                                                                  chain_1_length_probabilities={10: 1},
                                                                  chain_2_length_probabilities={10: 1},
                                                                  labels={"my_label": {"yes": 0.5, "no": 0.5},
                                                                          "unnecessary_label": {"yes": 0.5, "no": 0.5}},
                                                                  path=Path(tmp_path))
    elif dataset_type == "sequence":
        return RandomDatasetGenerator.generate_sequence_dataset(sequence_count=5,
                                                                length_probabilities={10: 1},
                                                                labels={"my_label": {"yes": 0.5, "no": 0.5},
                                                                        "unnecessary_label": {"yes": 0.5, "no": 0.5}},
                                                                path=Path(tmp_path))


def run_checks(parsed_args):
    logging.info(f"Testing new encoder file: '{parsed_args.encoder_file}'")

    assert os.path.isfile(parsed_args.encoder_file), f"Error: '{parsed_args.encoder_file}' is not a file"
    encoder_file_path = Path(parsed_args.encoder_file)

    check_encoder_file(encoder_file_path)
    encoder_class = get_encoder_class(encoder_file_path)

    default_params_filepath = get_default_params_filepath_from_name(encoder_file_path)
    default_params = check_default_args(default_params_filepath, parsed_args.no_default_parameters, class_name=encoder_file_path.stem)
    dummy_dataset = get_dummy_dataset(parsed_args.dataset_type, parsed_args.tmp_path)
    check_encoder_class(encoder_class, dummy_dataset, default_params, base_class_name=encoder_file_path.stem, tmp_path=parsed_args.tmp_path)

    logging.info("\n\nCongratulations! If the script ran this far without encountering any error, "
                 f"\nyou have successfully integrated a new DatasetEncoder for {parsed_args.dataset_type} datasets into immuneML. "
                 "\n\nTo finalise your encoder: "
                 "\n- Consider testing the encoder with other dataset types, if applicable."
                 f"\n- Check the exported design matrix to make sure the encoded data fulfills your expectations. This can be found at: '{parsed_args.tmp_path}/dme_report/design_matrix.csv' "
                 "\n- Please take a look if any warnings were thrown in this log file, and consider resolving those. These issues are not critical for the performance of your DatasetEncoder, but may be necessary to resolve before merging the code into the main immuneML branch."
                 "\n- Add class documentation containing a general description of the encoding, its parameters, and an example YAML snippet. "
                 "\n- Add a unit test for the new encoder. "
                 "\n\nFor more details, see https://docs.immuneml.uio.no/latest/developer_docs/how_to_add_new_encoding.html")



def main(args):
    parsed_args = parse_commandline_arguments(args)

    setup_logger(parsed_args.log_file)
    set_tmp_path(parsed_args.tmp_path)

    try:
        run_checks(parsed_args)
    except Exception as e:
        logging.error("\n\nA critical error occurred when testing the new DatasetEncoder:\n")
        logging.exception(e)

        raise e


if __name__ == "__main__":
    main(sys.argv[1:])
