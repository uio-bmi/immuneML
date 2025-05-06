import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.states.HPItem import HPItem
from immuneML.hyperparameter_optimization.states.TrainMLModelState import TrainMLModelState
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.train_ml_model_reports.PerformancePerLabel import PerformancePerLabel
from immuneML.util.ParameterValidator import ParameterValidator


class ConfusionMatrixPerLabel(PerformancePerLabel):
    """
    Report for TrainMLModel instruction: plots the confusion matrix split by the alternative label (for each label value).
    It can plot this on train or test data or for selection or assessment results. The confusion matrix will be
    reported per each hyperparameter setting (preprocessing, encoding and ML method combination).

    This report can be used only with TrainMLModel instruction under 'reports'.


    **Specification arguments:**

    - alternative_label (str): which label to use to stratify the data.

    - plot_on_train (bool): whether to plot the confusion matrix on the training data (False by default).

    - plot_on_test (bool): whether to plot the confusion matrix on the test data (True by default).

    - compute_for_selection (bool): whether to plot the confusion matrix on the selection data (False by default).

    - compute_for_assessment (bool): whether to plot the confusion matrix on the assessment data (True by default).



    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_conf_matrix:
                    ConfusionMatrixPerLabel:
                        alternative_label: "batch"
                        plot_on_train: False
                        plot_on_test: True
                        compute_for_selection: False
                        compute_for_assessment: True

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = cls.__name__
        valid_keys = ["plot_on_train", 'plot_on_test', 'compute_for_selection', 'compute_for_assessment',
                      'alternative_label', 'name']
        ParameterValidator.assert_keys(kwargs, valid_keys, location, location)
        return ConfusionMatrixPerLabel(**kwargs)

    def __init__(self, name: str = None, state: TrainMLModelState = None, plot_on_train: bool = False,
                 plot_on_test: bool = True,
                 compute_for_selection: bool = False, compute_for_assessment: bool = True,
                 alternative_label: str = None, result_path: Path = None):
        super().__init__(name=name, state=state, result_path=result_path, alternative_label=alternative_label,
                         number_of_processes=1,
                         metric='confusion_matrix', compute_for_assessment=compute_for_assessment,
                         compute_for_selection=compute_for_selection, plot_on_test=plot_on_test,
                         plot_on_train=plot_on_train)

    def _write_performance_tables(self, data: pd.DataFrame, dataset_desc: str, name_suffix: str, label_name: str):
        table_path = self.result_path / f"{name_suffix}_performance.csv"
        tmp_data = deepcopy(data)
        tmp_data['performance'] = tmp_data['performance'].astype(str)
        tmp_data.to_csv(table_path, index=False)
        return ReportOutput(table_path, self._get_desc_from_name_suffix(name_suffix, label_name, dataset_desc))

    def _process_dataset(self, dataset_info: dict, label: Label, hp_item: HPItem):
        predictions = dataset_info['predictions']
        metadata = dataset_info['metadata']

        # Calculate overall performance
        overall_performance = self._calculate_performance(predictions, label)

        # Calculate per-label performances and counts
        alt_label_performances = {}
        alt_label_counts = {}
        total_count = len(predictions)

        for alt_label_value in self.alternative_label_values:
            perf = self._get_performance_for_label_value(predictions, metadata, label, alt_label_value)
            count = self._get_count_for_label_value(metadata, alt_label_value)

            alt_label_performances[f"performance_{alt_label_value}"] = perf
            alt_label_counts[f"count_{alt_label_value}"] = count

        return {
            "setting": hp_item.hp_setting.get_key(),
            "performance": overall_performance,
            "example_count": total_count,
            **alt_label_performances,
            **alt_label_counts
        }

    def _create_performance_plot(self, data: pd.DataFrame, dataset_desc: str, name_suffix, label_name: str):
        figs = self._create_figure(data, label_name)
        outputs = []
        for i in range(len(figs)):
            plot_path = self.result_path / f"{name_suffix}_performance_plot_{i + 1}.html"
            figs[i].write_html(str(plot_path))
            outputs.append(ReportOutput(plot_path,
                                        self._get_desc_from_name_suffix(name_suffix, label_name, dataset_desc)))
        return outputs

    def _create_figure(self, data: pd.DataFrame, label_name: str):
        figs = []

        main_label = self.state.label_configuration.get_label_object(label_name)
        label_values = main_label.values

        for run_id in data['run_id'].unique():
            data_tmp = data[data['run_id'] == run_id]
            fig = make_subplots(rows=data_tmp['setting'].nunique(), cols=len(self.alternative_label_values) + 1,
                                subplot_titles=["all"] + [f"{self.alternative_label}: {alt_label_value}" for
                                                          alt_label_value in
                                                          self.alternative_label_values],
                                shared_yaxes=True, shared_xaxes=True, x_title="Predicted Label", y_title='True Label')

            for setting_ind, setting in enumerate(data_tmp['setting'].unique()):
                values = data_tmp[data_tmp['setting'] == setting]['performance'].values[0]
                fig.add_trace(go.Heatmap(texttemplate="%{text}",
                                         hovertemplate="True value: %{y}<br>Predicted value: %{x}<br>Count: %{z}<extra></extra>",
                                         z=values,
                                         x=label_values,
                                         y=label_values, showscale=False,
                                         text=values), row=setting_ind + 1, col=1)

                for ind, alt_label_value in enumerate(self.alternative_label_values):
                    values = data_tmp[data_tmp['setting'] == setting][f"performance_{alt_label_value}"].values[0]
                    fig.add_trace(go.Heatmap(z=values, texttemplate="%{text}", text=values,
                                             hovertemplate="True value: %{y}<br>Predicted value: %{x}<br>Count: %{z}<extra></extra>",
                                             x=label_values, y=label_values, showscale=False),
                                  row=setting_ind + 1, col=ind + 2)

                fig.update_yaxes(title_text=setting, row=setting_ind + 1, col=1)
                fig.update_layout(template='plotly_white', showlegend=False,
                                  margin_l=140, title=f"Confusion Matrix (split {run_id})")
                fig.update_annotations(selector=dict(text='True Label'), xshift=-100)

            figs.append(fig)

        return figs

    def _get_layout_settings(self, kwargs):
        return {
            **kwargs,
            "title": f"Performance by {self.alternative_label}",
            "xaxis_title": self.alternative_label,
            "yaxis_title": f"{self.metric.replace('_', ' ').title()}",
            "template": "plotly_white",
            "showlegend": True
        }

    def check_prerequisites(self):
        run_report = True

        if self.state is None:
            logging.warning(f"{self.__class__.__name__} can only be executed as a hyperparameter report. "
                            f"The report will not be created.")
            run_report = False

        if self.result_path is None:
            logging.warning(f"{self.__class__.__name__} requires an output 'path' to be set. "
                            f"The report will not be created.")
            run_report = False

        return run_report
