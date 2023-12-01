import copy

import pandas as pd

from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.util.Logger import print_log
from immuneML.workflows.steps.MLMethodTrainerParams import MLMethodTrainerParams
from immuneML.workflows.steps.Step import Step


class MLMethodTrainer(Step):

    @staticmethod
    def run(input_params: MLMethodTrainerParams = None):

        print_log(f"ML model training started...", include_datetime=True)

        method = MLMethodTrainer._fit_method(input_params)
        MLMethodTrainer.store(method, input_params)

        print_log(f"ML model training finished.", include_datetime=True)

        return method

    @staticmethod
    def _fit_method(input_params: MLMethodTrainerParams):
        method = copy.deepcopy(input_params.method)
        method.result_path = input_params.result_path

        if input_params.model_selection_cv:
            method.fit_by_cross_validation(encoded_data=input_params.dataset.encoded_data,
                                           number_of_splits=input_params.model_selection_n_folds,
                                           label=input_params.label,
                                           cores_for_training=input_params.cores_for_training,
                                           optimization_metric=input_params.optimization_metric)
        else:
            method.fit(encoded_data=input_params.dataset.encoded_data,
                       label=input_params.label,
                       cores_for_training=input_params.cores_for_training,
                       optimization_metric=input_params.optimization_metric)

        return method

    @staticmethod
    def store(method: MLMethod, input_params: MLMethodTrainerParams):
        method.store(input_params.result_path, input_params.dataset.encoded_data.feature_names, input_params.ml_details_path)
        train_predictions = method.predict(input_params.dataset.encoded_data, input_params.label)
        train_proba_predictions = method.predict_proba(input_params.dataset.encoded_data, input_params.label)

        df = pd.DataFrame({"example_ids": input_params.dataset.encoded_data.example_ids,
                           f"{input_params.label.name}_predicted_class": train_predictions[input_params.label.name],
                           f"{input_params.label.name}_true_class": input_params.dataset.encoded_data.labels[input_params.label.name]})

        for cls in method.get_classes():
            tmp = train_proba_predictions[input_params.label.name][cls] if train_proba_predictions is not None and train_proba_predictions[input_params.label.name] is not None else None
            df[f'{input_params.label.name}_{cls}_proba'] = tmp

        df.to_csv(input_params.train_predictions_path, index=False)
