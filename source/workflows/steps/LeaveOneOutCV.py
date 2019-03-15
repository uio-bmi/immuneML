# quality: peripheral

import copy
import json

from source.data_model.dataset.Dataset import Dataset
from source.encodings.EncoderParams import EncoderParams
from source.environment.LabelConfiguration import LabelConfiguration
from source.environment.MetricType import MetricType
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
            print("Iteration {} out of {}".format(i, input_params["dataset"].get_repertoire_count()))
            train_dataset, test_dataset = LeaveOneOutCV.split_dataset(input_params["dataset"], i)
            train_dataset, test_dataset = LeaveOneOutCV.encode_dataset(train_dataset, test_dataset, input_params, i, label_config)
            trained_methods = LeaveOneOutCV.train_methods(train_dataset, input_params, i)
            assessment = LeaveOneOutCV.assess(trained_methods, test_dataset, input_params, i, label_config)
            assessments.append(assessment)

        errors = LeaveOneOutCV.extract_errors(input_params["dataset"], assessments, input_params["metrics"])

        final_assessment = LeaveOneOutCV.assess_all(assessments, input_params["metrics"], input_params["results_aggregator"])

        with open(input_params["result_path"] + "assessments.json", "w") as file:
            json.dump(final_assessment, file, indent=2)

        with open(input_params["result_path"] + "errors.json", "w") as file:
            json.dump(errors, file, indent=2)

        return final_assessment

    @staticmethod
    def extract_errors(dataset: Dataset, assessments: list, metrics) -> dict:
        if MetricType.BALANCED_ACCURACY not in metrics and MetricType.ACCURACY not in metrics:
            return {}
        else:
            metric = MetricType.BALANCED_ACCURACY if MetricType.BALANCED_ACCURACY in metrics else MetricType.ACCURACY
            errors = {}
            random_performance = {label: 1.0 / len(dataset.params[label]) for label in dataset.params.keys()}
            for index, rep in enumerate(dataset.get_data()):
                if any([any([assessments[index][method][label][metric.name] <= random_performance[label] for label in assessments[index][method].keys()]) for method in assessments[index].keys()]):
                    errors[str(index) + "_" + rep.identifier] = {
                        "status": rep.metadata.custom_params,
                        "assessments": assessments[index],
                        "other_info": {
                            "chains": list({seq.metadata.chain if seq.metadata else None for seq in rep.sequences})
                        }
                    }
            return errors

    @staticmethod
    def average_assessment(assessments: list, metrics: list):
        res = {method_name:
                   {label:
                        {metric.name: 0 for metric in metrics}
                    for label in assessments[0][method_name].keys()}
               for method_name in assessments[0].keys()}

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
                "result_path": input_params["result_path"] + "loocv{}/".format(index) + input_params["encoder"].__name__ + "/ml_methods/",
                "dataset": dataset,
                "labels": input_params["labels"],
                "number_of_splits": input_params["cv"] if "cv" in input_params else 0,
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
                label_config.add_label(label, list(input_params["dataset"].params[label]))

        return label_config

    @staticmethod
    def split_dataset(dataset: Dataset, index):
        test = Dataset(filenames=[dataset.filenames[index]], params=dataset.params)
        train = Dataset(filenames=copy.deepcopy(dataset.filenames), params=dataset.params)
        del train.filenames[index]
        return train, test
