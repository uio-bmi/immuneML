# quality: peripheral

import copy

from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.environment.LabelConfiguration import LabelConfiguration
from source.workflows.steps.DataEncoder import DataEncoder
from source.workflows.steps.MLMethodAssessment import MLMethodAssessment
from source.workflows.steps.MLMethodTrainer import MLMethodTrainer
from source.workflows.steps.Step import Step


class LeaveOneOutCV(Step):

    @staticmethod
    def run(input_params: dict = None):
        return LeaveOneOutCV.perform_step(input_params)

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        pass

    @staticmethod
    def perform_step(input_params: dict = None):

        assessments = []

        label_config = LeaveOneOutCV.build_label_configuration(input_params)

        for i in range(input_params["dataset"].get_repertoire_count()):
            train_dataset, test_dataset = LeaveOneOutCV.split_dataset(input_params["dataset"], i)
            train_dataset, test_dataset = LeaveOneOutCV.encode_dataset(train_dataset, test_dataset, input_params, i, label_config)
            trained_methods = LeaveOneOutCV.train_methods(train_dataset, input_params, i)
            assessment = LeaveOneOutCV.assess(trained_methods, test_dataset, input_params, i, label_config)
            assessments.append(assessment)

        final_assessment = LeaveOneOutCV.assess_all(assessments, input_params["metrics"], input_params["results_aggregator"])

        return final_assessment

    @staticmethod
    def average_assessment(assessments: list, metrics: list):
        res = {
        method_name: {label: {metric.name: 0 for metric in metrics} for label in assessments[0][method_name].keys()} for
        method_name in assessments[0].keys()}
        for item in assessments:
            for metric in metrics:
                for method_name in item:
                    for label in item[method_name].keys():
                        res[method_name][label][metric.name] += item[method_name][label][metric.name]

        for metric in metrics:
            for method_name in assessments[0].keys():
                for label in assessments[0][method_name].keys():
                    res[method_name][label][metric.name] /= len(assessments)
        return res

    @staticmethod
    def assess_all(assessments: list, metrics: list, results_aggregator: str):
        assert len(assessments) > 0, "LeaveOneOutCV: There are no assessments to combine."

        if results_aggregator == "average":
            res = LeaveOneOutCV.average_assessment(assessments, metrics)
            return res
        else:
            raise NotImplementedError

    @staticmethod
    def assess(methods: list, dataset: Dataset, input_params: dict, index: int, label_config: LabelConfiguration) -> dict:
        assessment = MLMethodAssessment.run({
            "methods": methods,
            "dataset": dataset,
            "metrics": input_params["metrics"],
            "labels": input_params["labels"],
            "label_configuration": label_config,
            "predictions_path": input_params["result_path"] + "loocv{}/".format(index) + input_params["encoder"].__name__ + "/predictions/",
        })
        return assessment

    @staticmethod
    def train_methods(dataset: Dataset, input_params: dict, index: int) -> list:

        trained_methods = []

        for method in input_params["methods"]:
            trained_method = MLMethodTrainer.run({
                "method": method,
                "result_path": input_params["result_path"] + "loocv{}/".format(index) + input_params["encoder"].__name__ + "/",
                "dataset": dataset,
                "labels": input_params["labels"],
                "number_of_splits": input_params["cv"],
                "fit_method": input_params["fit_method"] if "fit_method" in input_params else "cv" if input_params["cv"] > 0 else None
            })
            trained_methods.append(trained_method)

        return trained_methods

    @staticmethod
    def encode_dataset(train_dataset: Dataset, test_dataset: Dataset, input_params: dict, index: int, label_config: LabelConfiguration):

        path = input_params["result_path"] + "loocv{}/".format(index) + input_params["encoder"].__name__ + "/"

        encoded_train_dataset = DataEncoder.run({"dataset": train_dataset,
                                                  "encoder_params": EncoderParams(result_path=path + "train/",
                                                                               label_configuration=label_config,
                                                                               model=input_params["encoder_params"]["model"],
                                                                               learn_model=True,
                                                                               model_path=path + "train/",
                                                                               vectorizer_path=path + "train/",
                                                                               scaler_path=path + "train/",
                                                                               pipeline_path=path + "train/"),
                                                   "encoder": input_params["encoder"]
                                                  })

        encoded_test_dataset = DataEncoder.run({"dataset": test_dataset,
                                                 "encoder_params": EncoderParams(result_path=path + "test/",
                                                                             label_configuration=label_config,
                                                                             model=input_params["encoder_params"]["model"],
                                                                             learn_model=False,
                                                                             model_path=path + "train/",
                                                                             vectorizer_path=path + "train/",
                                                                             scaler_path=path + "train/",
                                                                             pipeline_path=path + "train/"),
                                                  "encoder": input_params["encoder"]
                                                 })

        return encoded_train_dataset, encoded_test_dataset

    @staticmethod
    def build_label_configuration(input_params) -> LabelConfiguration:

        label_config = LabelConfiguration()

        for label in input_params["labels"]:
            if label in input_params["dataset"].params.keys():
                label_config.add_label(label, input_params["dataset"].params[label])

        return label_config

    @staticmethod
    def split_dataset(dataset: Dataset, index):
        test = Dataset(filenames=[dataset.filenames[index]], params=dataset.params)
        train = Dataset(filenames=copy.deepcopy(dataset.filenames), params=dataset.params)
        del train.filenames[index]
        return train, test
