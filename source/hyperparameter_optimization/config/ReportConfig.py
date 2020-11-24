from scripts.specification_util import update_docs_per_mapping


class ReportConfig:
    """
    A class encapsulating different report lists which can be executed while performing nested cross-validation (CV) using TrainMLModel
    instruction. All arguments are optional.

    Arguments:

        data (dict): :ref:`Data reports` to be executed on the whole dataset before it is split to training/test or training/validation

        data_splits (dict): :ref:`Data reports` to be executed after the data has been split into training and test (assessment CV loop) or training and validation (selection CV loop) datasets before they are encoded

        models (dict): :ref:`ML model reports` to be executed on all trained classifiers

        encoding (dict): :ref:`Encoding reports` to be executed on each of the encoded training/test datasets or training/validation datasets

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        # as a part of a TrainMLModel instruction, defining the outer (assessment) loop of nested cross-validation:
        assessment: # outer loop of nested CV
            split_strategy: random # perform Monte Carlo CV (randomly split the data into train and test)
            split_count: 5 # how many train/test datasets to generate
            training_percentage: 0.7 # what percentage of the original data should be used for the training set
            reports: # reports to execute on training/test datasets, encoded datasets and trained ML methods
                data_splits: # list of reports to execute on training/test datasets (before they are preprocessed and encoded)
                    - my_data_split_report
                encoding: # list of reports to execute on encoded training/test datasets
                    - my_encoding_report

        # as a part of a TrainMLModel instruction, defining the inner (selection) loop of nested cross-validation:
        selection: # inner loop of nested CV
            split_strategy: random # perform Monte Carlo CV (randomly split the data into train and validation)
            split_count: 5 # how many train/validation datasets to generate
            training_percentage: 0.7 # what percentage of the original data should be used for the training set
            reports: # reports to execute on training/validation datasets, encoded datasets and trained ML methods
                data_splits: # list of reports to execute on training/validation datasets (before they are preprocessed and encoded)
                    - my_data_split_report
                encoding: # list of reports to execute on encoded training/validation datasets
                    - my_encoding_report
                models:
                    - my_ml_model_report

    """

    def __init__(self, data_splits: dict = None, models: dict = None, data: dict = None, encoding: dict = None):

        self.data_split_reports = data_splits if data_splits is not None else {}
        self.encoding_reports = encoding if encoding is not None else {}
        self.model_reports = models if models is not None else {}
        self.data_reports = data if data is not None else {}

    @staticmethod
    def get_documentation():
        doc = str(ReportConfig.__doc__)
        mapping = {
            "data_splits (dict)": "data_splits",
            "models (dict)": "models",
            "data (dict)": "data",
            "encoding (dict)": "encoding"
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc
