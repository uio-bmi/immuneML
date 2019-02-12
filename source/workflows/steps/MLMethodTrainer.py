import numpy as np

from source.data_model.dataset.Dataset import Dataset
from source.ml_methods.MLMethod import MLMethod
from source.workflows.steps.Step import Step


class MLMethodTrainer(Step):

    @staticmethod
    def run(input_params: dict = None):
        MLMethodTrainer.check_prerequisites(input_params)
        method = MLMethodTrainer.perform_step(input_params)
        return method

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        assert input_params is not None, "MLMethodTrainer: input_params cannot be None."
        assert "method" in input_params and isinstance(input_params["method"], MLMethod), "MLMethodTrainer: an instance of MLMethod has to be passed in to input_params as parameter 'method'."
        assert "result_path" in input_params, "MLMethodTrainer: the result_path parameter is not set."
        assert "dataset" in input_params and isinstance(input_params["dataset"], Dataset), "MLMethodTrainer: the dataset parameter has to be set and has to contain an instance of Dataset object with encoded_data parameter also set."
        assert "labels" in input_params, "MLMethodTrainer: the labels parameter has to be set to a list of all labels which should be used for creating ML models."

    @staticmethod
    def perform_step(input_params: dict = None):
        method = input_params["method"]

        if not method.check_if_exists(input_params["result_path"]):
            method = MLMethodTrainer.__fit_method(input_params)
            method.store(input_params["result_path"])
        else:
            method.load(input_params["result_path"])

        return method

    @staticmethod
    def __fit_method(input_params: dict):
        X = input_params["dataset"].encoded_data["repertoires"]
        y = MLMethodTrainer.__filter_labels(input_params)
        parameter_grid = input_params["parameter_grid"] if "parameter_grid" in input_params else None
        method = input_params["method"]
        input_params["labels"].sort()

        if "fit_method" not in input_params or input_params["fit_method"] == "cv":
            method.fit_by_cross_validation(X=X, y=y,
                                           number_of_splits=input_params["number_of_splits"],
                                           parameter_grid=parameter_grid,
                                           label_names=input_params["labels"])
        else:
            method.fit(X, y)

        return method

    @staticmethod
    def __filter_labels(input_params: dict):

        label_names = input_params["dataset"].encoded_data["label_names"]
        y = input_params["dataset"].encoded_data["labels"].copy()

        for index, label in enumerate(label_names):
            if label not in input_params["labels"]:  # if the user did not specify that ML model should be built for this label
                y = np.delete(y, index, 0)

        return y
