import copy
import datetime

import pandas as pd

from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams
from immuneML.workflows.steps.Step import Step


class MLMethodTrainer(Step):

    @staticmethod
    def run(input_params: MLMethodTrainerParams = None):

        print(f"{datetime.datetime.now()}: ML model training started...", flush=True)

        method = MLMethodTrainer._fit_method(input_params)
        MLMethodTrainer.store(method, input_params)

        print(f"{datetime.datetime.now()}: ML model training finished.", flush=True)

        return method

    @staticmethod
    def _fit_method(input_params: MLMethodTrainerParams):
        method = copy.deepcopy(input_params.method)
        method.result_path = input_params.result_path

        if input_params.model_selection_cv:
            method.fit_by_cross_validation(encoded_data=input_params.dataset.encoded_data,
                                           number_of_splits=input_params.model_selection_n_folds,
                                           label_name=input_params.label,
                                           cores_for_training=input_params.cores_for_training,
                                           optimization_metric=input_params.optimization_metric)
        else:
            method.fit(encoded_data=input_params.dataset.encoded_data, label_name=input_params.label, cores_for_training=input_params.cores_for_training)

        return method

    @staticmethod
    def store(method: MLMethod, input_params: MLMethodTrainerParams):
        method.store(input_params.result_path, input_params.dataset.encoded_data.feature_names, input_params.ml_details_path)
        train_predictions = method.predict(input_params.dataset.encoded_data, input_params.label)
        train_proba_predictions = method.predict_proba(input_params.dataset.encoded_data, input_params.label)

        df = pd.DataFrame({"example_ids": input_params.dataset.encoded_data.example_ids,
                           f"{input_params.label}_predicted_class": train_predictions[input_params.label],
                           f"{input_params.label}_true_class": input_params.dataset.encoded_data.labels[input_params.label]})

        classes = method.get_classes_for_label(input_params.label)
        for cls_index, cls in enumerate(classes):
            tmp = train_proba_predictions[input_params.label][:, cls_index] if train_proba_predictions is not None and train_proba_predictions[input_params.label] is not None else None
            df["{}_{}_proba".format(input_params.label, cls)] = tmp

        df.to_csv(input_params.train_predictions_path, index=False)
