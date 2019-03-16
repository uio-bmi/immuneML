from source.data_model.dataset.Dataset import Dataset
from source.dsl_parsers.Parser import Parser
from source.encodings.EncoderParams import EncoderParams
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
from source.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.DataSplitter import DataSplitter
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.SignalImplanter import SignalImplanter


class Quickstart:

    @staticmethod
    def perform_analysis(params: dict):
        params = Quickstart.preprocess_params(params)
        dataset = Quickstart.generate_dataset(params)
        dataset = Quickstart.implant_signal(params, dataset)
        train, test = Quickstart.split_dataset(params, dataset)
        train, test = Quickstart.encode_datasets(params, train, test)
        methods = Quickstart.train_ml_algorithms(params, train)
        results = Quickstart.assess_ml_algorithms(methods, test, params)
        print(results)

    @staticmethod
    def preprocess_params(params: dict) -> dict:
        processed_params = Parser.parse(params)

        lc = LabelConfiguration()
        for signal in processed_params["signals"]:
            lc.add_label(signal.id, [True, False])
        processed_params["label_configuration"] = lc

        return processed_params

    @staticmethod
    def generate_dataset(params: dict) -> Dataset:
        print("#### generating dataset....")

        dataset = RandomDatasetGenerator.generate_dataset(repertoire_count=params["repertoire_count"],
                                                          sequence_count=params["sequence_count"],
                                                          path=params["result_path"])

        print("#### dataset generated....")

        return dataset

    @staticmethod
    def implant_signal(params: dict, dataset: Dataset) -> Dataset:
        print("#### implanting signal....")

        new_dataset = SignalImplanter.run({
            "repertoire_count": params["repertoire_count"],
            "sequence_count": params["sequence_count"],
            "simulation": params["simulation"],
            "signals": params["signals"],
            "result_path": params["result_path"] + "_".join([signal.id for signal in params["signals"]]) + "/",
            "dataset": dataset,
            "batch_size": params["batch_size"]
        })

        print("#### signal implanted....")
        return new_dataset

    @staticmethod
    def split_dataset(params: dict, dataset: Dataset):
        print("#### splitting dataset....")

        train_dataset, test_dataset = DataSplitter.run({
            "dataset": dataset,
            "training_percentage": params["training_percentage"]
        })

        print("#### dataset split....")
        return train_dataset, test_dataset

    @staticmethod
    def encode_datasets(params: dict, train_dataset: Dataset, test_dataset: Dataset):
        print("#### encoding datasets....")

        path = params["result_path"] + params["encoder"].__class__.__name__ + "/"

        encoded_train_dataset = DataEncoder.run({
            "dataset": train_dataset,
            "encoder": params["encoder"],
            "encoder_params": EncoderParams(
                model=params["encoder_params"],
                result_path=path + "train/",
                model_path=path,
                vectorizer_path=path,
                scaler_path=path,
                pipeline_path=path,
                batch_size=params["batch_size"],
                label_configuration=params["label_configuration"]
            )
        })

        encoded_test_dataset = DataEncoder.run({
            "dataset": test_dataset,
            "encoder": params["encoder"],
            "encoder_params": EncoderParams(
                model=params["encoder_params"],
                result_path=path + "test/",
                model_path=path,
                vectorizer_path=path,
                scaler_path=path,
                pipeline_path=path,
                batch_size=params["batch_size"],
                learn_model=False,
                label_configuration=params["label_configuration"]
            )
        })

        print("#### datasets encoded....")

        return encoded_train_dataset, encoded_test_dataset

    @staticmethod
    def train_ml_algorithms(params: dict, train_dataset: Dataset):
        print("#### training ML algorithms....")

        methods = []

        for method in params["ml_methods"]:
            trained_method = MLMethodTrainer.run({
                "method": method,
                "result_path": params["result_path"] + params["encoder"].__class__.__name__ + "/ml_methods/",
                "dataset": train_dataset,
                "labels": params["label_configuration"].get_labels_by_name(),
                "number_of_splits": params["cv"]
            })

            methods.append(trained_method)

        print("#### ML algorithms trained....")

        return methods

    @staticmethod
    def assess_ml_algorithms(methods: list, test_dataset: Dataset, params: dict) -> dict:
        print("#### assessing performance....")

        results = MLMethodAssessment.run({
            "methods": methods,
            "dataset": test_dataset,
            "metrics": [MetricType.BALANCED_ACCURACY],
            "labels": params["label_configuration"].get_labels_by_name(),
            "predictions_path": params["result_path"] + params["encoder"].__class__.__name__ + "/predictions/",
            "label_configuration": params["label_configuration"]
        })

        print("#### performance assessed....")

        return results


Quickstart.perform_analysis({
    "repertoire_count": 400,
    "sequence_count": 500,
    "receptor_type": "TCR",
    "result_path": EnvironmentSettings.root_path + "simulation_results/",
    "ml_methods": ["RandomForestClassifier"],  # other classifiers: "LogisticRegression", "SVM"
    "training_percentage": 0.7,
    "cv": 10,
    "encoder": "KmerFrequencyEncoder",
    "encoder_params": {
        "sequence_encoding": "continuous_kmer",
        "k": 3,
        "reads": "unique",
        "normalization_type": "relative_frequency"
    },
    "simulation": {
        "motifs": [
            {
                "id": "motif1",
                "seed": "CAS",
                "instantiation": "identity"
            }
        ],
        "signals": [
            {
                "id": "signal1",
                "motifs": ["motif1"],
                "implanting": "healthy_sequences"
            }
        ],
        "implanting": [{
            "signals": ["signal1"],
            "repertoires": 0.5,
            "sequences": 0.2
        }]
    }
})
