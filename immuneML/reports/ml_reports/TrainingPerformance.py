import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn import metrics as sklearn_metrics
from sklearn.preprocessing import label_binarize

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization import HPSetting
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.ml_metrics import ml_metrics
from immuneML.ml_metrics.Metric import Metric
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.MLReport import MLReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class TrainingPerformance(MLReport):
    """
    A report that plots the evaluation metrics for the performance given machine learning model and training dataset.
    The available metrics are accuracy, balanced_accuracy, confusion_matrix, f1_micro, f1_macro, f1_weighted, precision,
    recall, auc and log_loss (see :py:obj:`immuneML.environment.Metric.Metric`).

    Arguments:

        metrics (list): A list of metrics used to evaluate training performance. See :py:obj:`immuneML.environment.Metric.Metric` for available options.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_performance_report:
            TrainingPerformance: 
                metrics:
                    - accuracy
                    - balanced_accuracy
                    - confusion_matrix
                    - f1_micro
                    - f1_macro
                    - f1_weighted
                    - precision
                    - recall
                    - auc
                    - log_loss

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = "TrainingPerformance"        
        valid_metrics = [m.name for m in Metric]

        name = kwargs["name"] if "name" in kwargs else None
        metrics = kwargs["metrics"] if "metrics" in kwargs else valid_metrics
        metrics = [m.upper() for m in metrics]
        
        ParameterValidator.assert_all_in_valid_list(metrics, valid_metrics, location, 'metrics')

        return TrainingPerformance(set(metrics), name=name)

    def __init__(self, metrics: set, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting: HPSetting = None):
        super().__init__(train_dataset, test_dataset, method, result_path, name, hp_setting)
        self.metrics_set = set(metrics)

    def _generate(self) -> ReportResult:
        
        X = self.train_dataset.encoded_data
        predicted_y = self.method.predict(X, self.label)[self.label]
        predicted_proba_y = self.method.predict_proba(X, self.label)[self.label]
        true_y = self.train_dataset.encoded_data.labels[self.label]
        classes = self.method.get_classes()

        PathBuilder.build(self.result_path)

        scores = {}
        output = {
            'tables': [],
            'figures': []
        }

        for metric in self.metrics_set:
            _score = TrainingPerformance._compute_score(
                Metric[metric],
                predicted_y,
                predicted_proba_y,
                true_y,
                classes,
            )
            if metric == 'CONFUSION_MATRIX':
                self._generate_heatmap(classes, classes, _score, metric, output)
            else:
                scores[metric] = _score

        scores_df = pd.DataFrame.from_dict(scores, orient='index')
        scores_df.columns = [self.label]

        self._generate_barplot(scores_df, output)

        return ReportResult(self.name,
                            output_tables=output['tables'],
                            output_figures=output['figures'])

    @staticmethod
    def _compute_score(metric: Metric, predicted_y, predicted_proba_y, true_y, labels):
        if hasattr(ml_metrics, metric.value):
            fn = getattr(ml_metrics, metric.value)
        else:
            fn = getattr(sklearn_metrics, metric.value)

        if hasattr(true_y, 'dtype') and true_y.dtype.type is np.str_ or isinstance(true_y, list) and any(isinstance(item, str) for item in true_y):
            true_y = label_binarize(true_y, classes=labels)
            predicted_y = label_binarize(predicted_y, classes=labels)

        try:
            if metric in Metric.get_probability_based_metric_types():
                predictions = predicted_proba_y
                if predicted_proba_y is None:
                    warnings.warn(f"TrainingPerformance: metric {metric} is specified, but the chosen ML method does not output "
                                  f"class probabilities. Using predicted classes instead...")
                    predictions = predicted_y
            else:
                predictions = predicted_y
            
            score = fn(true_y, predictions)

        except ValueError as err:
            warnings.warn(f"TrainingPerformance: score for metric {metric.name} could not be calculated."
                          f"\nPredicted values: {predicted_y}\nTrue values: {true_y}.\nMore details: {err}", RuntimeWarning)
            score = "not computed"

        return score

    def _generate_barplot(self, df, output):
        path_csv = self.result_path / f"{self.name}.csv"
        path_html = self.result_path / f"{self.name}.html"

        df.to_csv(path_csv)

        layout = go.Layout(title=f'Evaluation Metrics ({self.label})',
                           xaxis=dict(title='Value'),
                           yaxis=dict(title='Metrics'))
        trace = go.Bar(x=df[self.label], y=df.index, marker_color='navy', orientation='h')

        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(str(path_html))

        output['tables'].append(ReportOutput(path_csv, "TrainingPerformance table"))
        output['figures'].append(ReportOutput(path_html, "TrainingPerformance html"))

        return

    def _generate_heatmap(self, x, y, z, metric, output, xlabel='Prediction', ylabel='Ground Truth', zlabel='Count'):
        path_csv = self.result_path / f"{self.name}_{metric.lower()}.csv"
        path_html = self.result_path / f"{self.name}_{metric.lower()}.html"
        
        z_flip = np.flipud(z)

        hovertext = []
        for yi, yy in enumerate(y):
            hovertext.append(list())
            for xi, xx in enumerate(x):
                hovertext[-1].append(f"{xlabel}: {xx}<br />{ylabel}: {yy}<br />{zlabel}: {z_flip[yi][xi]}")

        layout = go.Layout(
            title=f'Evaluation: {metric} ({self.label})',
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel)
        )
        trace = go.Heatmap(
            z=z_flip,
            x=x,
            y=y,
            hoverongaps = False,
            colorscale = 'burgyl',
            hoverinfo='text',
            text=hovertext
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(str(path_html))

        z_df = pd.DataFrame(z)
        z_df.columns = f'{xlabel} (' + pd.Index(map(str, x)) + ')'
        z_df.index = f'{ylabel} (' +  pd.Index(map(str, y)) + ')'
        z_df.to_csv(path_csv)   
        
        output['tables'].append(ReportOutput(path_csv, f"TrainingPerformance table ({metric.lower()})"))
        output['figures'].append(ReportOutput(path_html, f"TrainingPerformance html ({metric.lower()})"))
        
        return

    def check_prerequisites(self) -> bool:
        if not hasattr(self, "result_path") or self.result_path is None:
            warnings.warn(f"{self.__class__.__name__} requires an output"
                          f" 'path' to be set. {self.__class__.__name__}"
                          f" report will not be created.")
            return False

        if self.train_dataset is None or self.train_dataset.encoded_data is None:
            warnings.warn(
                f"{self.__class__.__name__}: train dataset is"
                f" not encoded and can not be run."
                f"{self.__class__.__name__} report will not be created.")
            return False

        if self.method is None:
            warnings.warn(
                f"{self.__class__.__name__}: method is"
                f" not defined and can not be run."
                f"{self.__class__.__name__} report will not be created.")
            return False

        return True
