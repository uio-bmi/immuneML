import copy

from source.workflows.steps.Step import Step


class MLMethodTrainer(Step):

    @staticmethod
    def check_prerequisites(input_params: dict = None):
        pass

    @staticmethod
    def run(input_params: dict = None):
        MLMethodTrainer.check_prerequisites(input_params)
        method = MLMethodTrainer.perform_step(input_params)
        return method

    @staticmethod
    def perform_step(input_params: dict = None):
        method = input_params["method"]

        if not method.check_if_exists(input_params["result_path"]):
            method = MLMethodTrainer._fit_method(input_params)
            method.store(input_params["result_path"])
        else:
            method.load(input_params["result_path"])

        return method

    @staticmethod
    def _fit_method(input_params: dict):
        X = input_params["dataset"].encoded_data.repertoires
        y = MLMethodTrainer._filter_labels(input_params)
        method = input_params["method"]
        input_params["labels"].sort()

        if input_params["model_selection_cv"] is True:
            method.fit_by_cross_validation(X=X, y=y,
                                           number_of_splits=input_params["model_selection_n_folds"],
                                           label_names=input_params["labels"])
        else:
            method.fit(X, y, label_names=input_params["labels"])

        return method

    @staticmethod
    def _filter_labels(input_params: dict):

        y = copy.deepcopy(input_params["dataset"].encoded_data.labels)

        for label in input_params["dataset"].encoded_data.labels.keys():
            if label not in input_params["labels"]:  # if the user did not specify that ML model should be built for this label
                del y[label]

        return y
