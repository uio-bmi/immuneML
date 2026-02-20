from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering import clustering_runner
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringSetting, ClusteringItem


@dataclass
class ClItems:
    cl_setting: ClusteringSetting
    discovery: List[ClusteringItem] = field(default_factory=list)
    tuning: List[ClusteringItem] = field(default_factory=list)


@dataclass
class StabilityLange:
    """Class to run stability-based hyperparameter assessment for clustering algorithms, based on Lange et al. 2004.

    Reference:

    Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-Based Validation of Clustering Solutions.
    Neural Computation, 16(6), 1299–1323. https://doi.org/10.1162/089976604773717621

    """
    discovery_datasets: List[Dataset]
    tuning_datasets: List[Dataset]
    clustering_settings: List[ClusteringSetting]
    result_path: Path
    number_of_processes: int
    sequence_type: SequenceType = SequenceType.AMINO_ACID
    region_type: RegionType = RegionType.IMGT_CDR3
    clustering_items: Dict[str, ClItems] = None

    def run(self):

        performances = {}

        for cl_setting in self.clustering_settings:
            distances = self._compute_stability_for_setting(cl_setting)
            performances[cl_setting.get_key()] = distances

        report_result = self._create_report_from_distances(performances)
        return report_result, report_result.output_tables[0].path

    def _create_report_from_distances(self, performances: Dict[str, np.ndarray]) -> ReportResult:
        df = pd.DataFrame(performances)
        df['split_id'] = np.arange(1, df.shape[0] + 1)
        df.to_csv(self.result_path / 'adjusted_rand_score_per_cl_setting.csv', index=False)

        figure = self.make_figure(df)
        return ReportResult(output_figures=[figure],
                            output_tables=[ReportOutput(self.result_path / 'adjusted_rand_score_per_cl_setting.csv',
                                                        name="Normalized Distances per Clustering Setting")],
                            name="Clustering Stability Analysis",
                            info="Report on clustering stability analysis based on agreement between clusterings on "
                                 "discovery and tuning datasets. The clusterings are performed separately on the two "
                                 "datasets, then the clustering from discovery dataset is transferred in a supervised "
                                 "manner to the tuning dataset and the adjusted Rand score between the two clusterings on the "
                                 "tuning set is reported.")

    def make_figure(self, df: pd.DataFrame) -> ReportOutput:
        import plotly.express as px
        df_long = df[[col for col in df.columns if col != 'split_id']].melt(var_name='clustering_setting', value_name='adjusted_rand_score')
        fig = px.box(df_long, x='clustering_setting', y='adjusted_rand_score', points='all', color='clustering_setting',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(xaxis_title="clustering setting", showlegend=False,
                          yaxis_title='adjusted Rand score', template="plotly_white")
        fig.update_traces(marker=dict(opacity=0.75), jitter=0.3)

        plot_path = PlotlyUtil.write_image_to_file(fig, self.result_path / f"stability_boxplot.html", df.shape[0])

        return ReportOutput(plot_path, name=f"Clustering Stability Analysis")

    def _compute_stability_for_setting(self, cl_setting: ClusteringSetting) -> np.ndarray:
        cl_items = ClItems(cl_setting=cl_setting)
        distances = np.zeros(len(self.discovery_datasets))
        unique_clusters = set()
        setting_path = self.result_path / cl_setting.get_key()

        for i in range(len(self.discovery_datasets)):
            clr_disc, _ = clustering_runner.run_setting(dataset=self.discovery_datasets[i], cl_setting=cl_setting,
                                                        path=setting_path / 'discovery', run_id=i, evaluate=False,
                                                        number_of_processes=self.number_of_processes,
                                                        sequence_type=self.sequence_type, region_type=self.region_type,
                                                        predictions_df=None, metrics=[],
                                                        label_config=LabelConfiguration())

            unique_clusters = set(clr_disc.item.predictions).union(unique_clusters)

            cl_items.discovery.append(clr_disc.item)

            clr, _ = clustering_runner.run_setting(dataset=self.tuning_datasets[i], cl_setting=cl_setting,
                                                   path=setting_path / 'tuning', run_id=i, evaluate=False,
                                                   number_of_processes=self.number_of_processes,
                                                   sequence_type=self.sequence_type, region_type=self.region_type,
                                                   predictions_df=None, metrics=[], label_config=LabelConfiguration())
            cl_items.tuning.append(clr.item)

            cl_item_transferred = self.transfer_clustering(clr_disc.item, dataset=self.tuning_datasets[i],
                                                           path=setting_path,
                                                           run_id=i)

            distances[i] = adjusted_rand_score(clr.item.predictions, cl_item_transferred.predictions)

        return distances

    def transfer_clustering(self, cl_item: ClusteringItem, dataset: Dataset, path: Path, run_id: int) -> ClusteringItem:
        classifier = clustering_runner.train_cluster_classifier(cl_item)

        # Apply classifier to validation data
        applied_cl_item = clustering_runner.apply_cluster_classifier(dataset=dataset, cl_setting=cl_item.cl_setting,
                                                                     classifier=classifier, encoder=cl_item.encoder,
                                                                     dim_red_method=cl_item.dim_red_method,
                                                                     predictions_path=PathBuilder.build(path) / f'predictions_run_{run_id + 1}_tuning_transferred.csv',
                                                                     number_of_processes=self.number_of_processes,
                                                                     sequence_type=self.sequence_type,
                                                                     region_type=self.region_type)

        return applied_cl_item

    def _fit_and_predict(self, dataset: Dataset, method: ClusteringMethod) -> np.ndarray:
        """Fit clustering method and get predictions."""
        if hasattr(method, 'fit_predict'):
            return method.fit_predict(dataset)
        else:
            method.fit(dataset)
            return method.predict(dataset)
