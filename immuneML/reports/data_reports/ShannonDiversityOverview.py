import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.diversity_encoding.ShannonDiversityEncoder import ShannonDiversityEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class ShannonDiversityOverview(DataReport):
    """
    Computes Shannon diversity for each repertoire using Shannon diversity encoder and plots the
    results in a histogram, optionally stratified by labels.

    **Dataset type:**

    - Repertoire Dataset

    **Specification arguments:**

    - color_label (str): The label used to color the histogram bars. Default is None.

    - facet_row_label (str): The label used to facet the histogram into multiple rows.
      Default is None, meaning no row faceting.

    - facet_col_label (str): The label used to facet the histogram into multiple columns.
      Default is None, meaning no column faceting.

     **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                shannon_div_rep:
                    ShannonDiversityOverview:
                        color_label: disease_status


    """

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1, color_label: str = None, facet_row_label: str = None,
                 facet_col_label: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.color_label = color_label
        self.facet_row_label = facet_row_label
        self.facet_col_label = facet_col_label

    @classmethod
    def build_object(cls, **kwargs):
        return ShannonDiversityOverview(**kwargs)

    def check_prerequisites(self) -> bool:
        valid = isinstance(self.dataset, RepertoireDataset)

        if not valid:
            logging.warning(f"ShannonDiversityOverview: Dataset must be of type RepertoireDataset, "
                            f"but got {type(self.dataset)}.")

        return valid

    def _generate(self) -> ReportResult:
        encoded_dataset = (ShannonDiversityEncoder.build_object(self.dataset)
                           .encode(self.dataset, EncoderParams(self.result_path, encode_labels=False)))

        PathBuilder.build(self.result_path)

        df, table_output = self.prepare_data(encoded_dataset)

        figure_output = self._safe_plot(encoded_df=df)

        return ReportResult(name=self.name, info="Shannon diversity per repertoire",
                            output_figures=[figure_output],
                            output_tables=[table_output])

    def prepare_data(self, encoded_dataset) -> Tuple[pd.DataFrame, ReportOutput]:
        labels = ['subject_id'] if 'subject_id' in self.dataset.labels.keys() else []
        for label in [self.color_label, self.facet_row_label, self.facet_col_label]:
            if label is not None:
                labels.append(label)

        df = pd.DataFrame({'shannon_diversity': encoded_dataset.encoded_data.examples,
                           'repertoire_id': encoded_dataset.get_example_ids(),
                           **self.dataset.get_metadata(labels)})

        df.sort_values(by='shannon_diversity', ascending=False, inplace=True)

        df.to_csv(self.result_path / 'shannon_diversity.csv', index=False)

        return df, ReportOutput(self.result_path / 'shannon_diversity.csv', name='Shannon diversity')

    def _plot(self, encoded_df) -> ReportOutput:

        facet_labels = []
        if self.facet_row_label:
            facet_labels.append(self.facet_row_label)
        if self.facet_col_label:
            facet_labels.append(self.facet_col_label)

        encoded_df['repertoire_index'] = encoded_df.groupby(facet_labels).cumcount() \
            if len(facet_labels) > 0 else list(range(encoded_df.shape[0]))

        hover_data_cols = ['repertoire_id'] + (['subject_id'] if 'subject_id' in encoded_df.columns else [])

        fig = px.bar(encoded_df, x='repertoire_index', y='shannon_diversity', facet_row=self.facet_row_label,
                     color=self.color_label, title='Shannon diversity per repertoire', facet_col=self.facet_col_label,
                     color_discrete_sequence=px.colors.diverging.Tealrose, hover_data=hover_data_cols)
        fig.update_layout(template="plotly_white", yaxis_title='Shannon diversity',
                          xaxis_title='Repertoires sorted by Shannon diversity')

        fig.update_traces(
            hovertemplate=(
                "Repertoire id: %{customdata[0]}<br>" +
                "Subject id: %{customdata[1]}<br>" if 'subject_id' in encoded_df.columns else "" +
                "Shannon diversity: %{y}<extra></extra>"
            )
        )

        fig.write_html(str(self.result_path / 'shannon_diversity.html'))

        return ReportOutput(self.result_path / 'shannon_diversity.html', name='Shannon diversity')
