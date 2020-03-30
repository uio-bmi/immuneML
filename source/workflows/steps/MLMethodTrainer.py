import copy

import pandas as pd

from source.ml_methods.MLMethod import MLMethod
from source.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams
from source.workflows.steps.Step import Step


class MLMethodTrainer(Step):

    @staticmethod
    def run(input_params: MLMethodTrainerParams = None):
        method = copy.deepcopy(input_params.method)

        if not method.check_if_exists(input_params.result_path):
            method = MLMethodTrainer._fit_method(input_params)
            MLMethodTrainer.store(method, input_params)
        else:
            method.import_dataset(input_params.result_path)

        return method

    @staticmethod
    def _fit_method(input_params: MLMethodTrainerParams):
        X = input_params.dataset.encoded_data.examples
        y = MLMethodTrainer._filter_labels(input_params)
        method = input_params.method

        if input_params.model_selection_cv:
            method.fit_by_cross_validation(X=X, y=y,
                                           number_of_splits=input_params.model_selection_n_folds,
                                           label_names=[input_params.label],
                                           cores_for_training=input_params.cores_for_training)
        else:
            method.fit(X, y, label_names=[input_params.label], cores_for_training=input_params.cores_for_training)

        return method

    @staticmethod
    def store(method: MLMethod, input_params: MLMethodTrainerParams):
        method.store(input_params.result_path, input_params.dataset.encoded_data.feature_names, input_params.ml_details_path)
        train_predictions = method.predict(input_params.dataset.encoded_data.examples, [input_params.label])
        train_proba_predictions = method.predict_proba(input_params.dataset.encoded_data.examples, [input_params.label])

        df = pd.DataFrame({"example_ids": input_params.dataset.encoded_data.example_ids,
                           f"{input_params.label}_predicted_class": train_predictions[input_params.label],
                           f"{input_params.label}_true_class": input_params.dataset.encoded_data.labels[input_params.label]})

        classes = method.get_classes_for_label(input_params.label)
        for cls_index, cls in enumerate(classes):
            tmp = train_proba_predictions[input_params.label][:, cls_index] if train_proba_predictions is not None and train_proba_predictions[input_params.label] is not None else None
            df["{}_{}_proba".format(input_params.label, cls)] = tmp

        df.to_csv(input_params.train_predictions_path, index=False)

    @staticmethod
    def _filter_labels(input_params: MLMethodTrainerParams):

        y = copy.deepcopy(input_params.dataset.encoded_data.labels)

        for label in input_params.dataset.encoded_data.labels.keys():
            if label != input_params.label:
                del y[label]

        return y
