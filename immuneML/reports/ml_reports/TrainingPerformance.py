import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.hyperparameter_optimization import HPSetting
from immuneML.ml_methods.classifiers.MLMethod import MLMethod
from immuneML.ml_metrics.ClassificationMetric import ClassificationMetric
from immuneML.ml_metrics.MetricUtil import MetricUtil
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

    Specification arguments:

    - metrics (list): A list of metrics used to evaluate training performance. See :py:obj:`immuneML.environment.Metric.Metric` for available options.


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
        valid_metrics = [m.name for m in ClassificationMetric]

        name = kwargs["name"] if "name" in kwargs else None
        metrics = kwargs["metrics"] if "metrics" in kwargs else valid_metrics
        metrics = [m.upper() for m in metrics]
        
        ParameterValidator.assert_all_in_valid_list(metrics, valid_metrics, location, 'metrics')

        return TrainingPerformance(set(metrics), name=name)

    def __init__(self, metrics: set, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting: HPSetting = None, label=None, number_of_processes: int = 1):
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, method=method, result_path=result_path,
                         name=name, hp_setting=hp_setting, label=label, number_of_processes=number_of_processes)
        self.metrics_set = set(metrics)

    def _generate(self) -> ReportResult:
        
        X = self.train_dataset.encoded_data
        predicted_y = self.method.predict(X, self.label)[self.label.name]
        predicted_proba_y = self.method.predict_proba(X, self.label)[self.label.name][self.label.positive_class]
        true_y = self.train_dataset.encoded_data.labels[self.label.name]
        classes = self.method.get_classes()
        example_weights = self.train_dataset.get_example_weights()

        PathBuilder.build(self.result_path)

        scores = {}
        output = {
            'tables': [],
            'figures': []
        }

        for metric in self.metrics_set:
            _score = MetricUtil.score_for_metric(metric=ClassificationMetric.get_metric(metric),
                                                 predicted_y=predicted_y, predicted_proba_y=predicted_proba_y,
                                                 true_y=true_y, example_weights=example_weights, classes=classes)

            if metric == 'CONFUSION_MATRIX':
                self._generate_heatmap(classes, classes, _score, metric, output)
            else:
                scores[metric] = _score

        scores_df = pd.DataFrame.from_dict(scores, orient='index')
        scores_df.columns = [self.label.name]

        self._generate_barplot(scores_df, output)

        return ReportResult(self.name,
                            info="Plots the evaluation metrics for the performance given machine learning model and training dataset.",
                            output_tables=output['tables'],
                            output_figures=output['figures'])

    def _generate_barplot(self, df, output):
        import plotly.express as px

        path_csv = self.result_path / f"{self.name}.csv"
        path_html = self.result_path / f"{self.name}.html"

        df.to_csv(path_csv)

        figure = px.bar(df, x=df.index, y=self.label.name, labels={'index': "metrics"},
                        template='plotly_white', color_discrete_sequence=px.colors.diverging.Tealrose,
                        title=f"Evaluation metrics ({self.label})")

        figure.write_html(str(path_html))

        output['tables'].append(ReportOutput(path_csv, "training performance in csv"))
        output['figures'].append(ReportOutput(path_html, "training performance on selected metrics"))

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
            warnings.warn(f"{self.__class__.__name__} requires an output 'path' to be set. {self.__class__.__name__}"
                          f" report will not be created.")
            return False

        if self.train_dataset is None or self.train_dataset.encoded_data is None:
            warnings.warn(
                f"{self.__class__.__name__}: train dataset is not encoded and can not be run."
                f"{self.__class__.__name__} report will not be created.")
            return False

        if self.method is None:
            warnings.warn(
                f"{self.__class__.__name__}: method is not defined and can not be run."
                f"{self.__class__.__name__} report will not be created.")
            return False

        return True
