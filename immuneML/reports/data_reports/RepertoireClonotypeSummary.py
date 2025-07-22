import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class RepertoireClonotypeSummary(DataReport):
    """
    Shows the number of distinct clonotypes per repertoire in a given dataset as a bar plot.

    **Specification arguments:**

    - color_label (str): the label to color the bar plot by (optional, default: None)

    - facet_label (str): the label to facet the bar plot by (optional, default: None)

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_clonotype_summary_rep:
                    RepertoireClonotypeSummary:
                        color_label: celiac
                        facet_label: hla


    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None, number_of_processes: int = 1,
                 color_label: str = None, facet_label: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.color_label = color_label
        self.facet_label = facet_label

    @classmethod
    def build_object(cls, **kwargs):
        return RepertoireClonotypeSummary(**kwargs)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        return self._safe_plot()

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        column_names = []
        if self.color_label:
            column_names.append(self.color_label)
        if self.facet_label:
            column_names.append(self.facet_label)

        if len(column_names) > 0:
            metadata = self.dataset.get_metadata(column_names, return_df=True)
            return pd.concat([df, metadata], axis=1)
        else:
            return df

    def _plot(self) -> ReportResult:
        clonotypes = pd.DataFrame({'clonotype_count': [self._get_clonotype_count(repertoire)
                                   for repertoire in self.dataset.get_data()]})

        clonotypes['repertoire_id'] = self.dataset.get_example_ids()
        clonotypes = self.add_labels(clonotypes)
        clonotypes.sort_values(by='clonotype_count', ascending=False, inplace=True)
        clonotypes['repertoire_index'] = clonotypes.groupby(self.facet_label).cumcount() if self.facet_label else list(range(clonotypes.shape[0]))

        fig = px.bar(clonotypes, x='repertoire_index', y='clonotype_count', facet_row=self.facet_label,
                     color=self.color_label, title='Clonotype count per repertoire',
                     color_discrete_sequence=px.colors.diverging.Tealrose)
        fig.update_layout(template="plotly_white", yaxis_title='clonotype count',
                          xaxis_title='repertoires')

        if self.facet_label:
            facet_label_counts = {str(k): v for k, v in clonotypes[self.facet_label].value_counts().to_dict().items()}
            for annotation in fig.layout.annotations:
                group_label = annotation.text
                if '=' in group_label:
                    group = group_label.split('=')[1]
                    count = facet_label_counts.get(group, 0)
                    annotation.text = f"{group_label}<br>(n={count})"

        fig.add_annotation(
            text="clonotype counts",  # Your y-axis label
            xref="paper", yref="paper",  # Use paper coordinates (0–1)
            x=-0.07, y=0.5,  # Position to the left and centered vertically
            showarrow=False,
            textangle=-90,  # Vertical orientation
            font=dict(size=14)
        )

        fig.update_layout(margin=dict(l=80))
        fig.update_yaxes(title='')

        clonotypes.to_csv(self.result_path / 'clonotype_count_per_repertoire.csv', index=False)
        fig.write_html(str(self.result_path / 'clonotype_count_per_repertoire.html'))

        return ReportResult(name=self.name, info="Clonotype count per repertoire",
                            output_figures=[ReportOutput(self.result_path / 'clonotype_count_per_repertoire.html',
                                            name='Clonotype count per repertoire')],
                            output_tables=[ReportOutput(self.result_path / 'clonotype_count_per_repertoire.csv',
                                                        name='Clonotype count per repertoire')])

    def _get_clonotype_count(self, repertoire: Repertoire) -> int:

        sequences = repertoire.data.topandas()

        sequence_count = sequences.shape[0]
        unique_sequence_count = len(sequences.groupby(['cdr3_aa', 'v_call', 'j_call']).size().reset_index(name='count'))
        if sequence_count != unique_sequence_count:
            logging.warning(f"{RepertoireClonotypeSummary.__name__}: {self.name}: for repertoire {repertoire.identifier}, "
                            f"there are {sequence_count} sequences, but {unique_sequence_count} unique (CDR3 amino acid"
                            f" sequence, V call, J call) combinations.")

        return unique_sequence_count

    def check_prerequisites(self) -> bool:
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            logging.warning(f"{RepertoireClonotypeSummary.__name__}: report can be generated only from "
                            f"RepertoireDataset. Skipping this report...")
            return False
