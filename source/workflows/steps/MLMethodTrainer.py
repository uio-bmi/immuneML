import copy

from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams
from source.workflows.steps.Step import Step


class MLMethodTrainer(Step):

    @staticmethod
    def run(input_params: MLMethodTrainerParams = None):
        method = copy.deepcopy(input_params.method)

        if not method.check_if_exists(input_params.result_path):
            method = MLMethodTrainer._fit_method(input_params)
            method.store(input_params.result_path, input_params.dataset.encoded_data.feature_names)
        else:
            method.load(input_params.result_path)

        return method

    @staticmethod
    def _fit_method(input_params: MLMethodTrainerParams):
        X = input_params.dataset.encoded_data.examples
        y = MLMethodTrainer._filter_labels(input_params)
        method = input_params.method
        input_params.labels.sort()

        if input_params.model_selection_cv:
            method.fit_by_cross_validation(X=X, y=y,
                                           number_of_splits=input_params.model_selection_n_folds,
                                           label_names=input_params.labels,
                                           cores_for_training=input_params.cores_for_training)
        else:
            method.fit(X, y, label_names=input_params.labels, cores_for_training=input_params.cores_for_training)

        return method

    @staticmethod
    def _filter_labels(input_params: MLMethodTrainerParams):

        y = copy.deepcopy(input_params.dataset.encoded_data.labels)

        for label in input_params.dataset.encoded_data.labels.keys():
            if label not in input_params.labels:
                del y[label]

        return y
