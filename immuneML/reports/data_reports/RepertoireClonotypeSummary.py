import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class RepertoireClonotypeSummary(DataReport):
    """
    Shows the number of distinct clonotypes per repertoire in a given dataset as a bar plot.

    **Specification arguments:**

    - color_by_label (str): name of the label to use to color the plot, e.g., could be disease label, or None

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_clonotype_summary_rep:
                    RepertoireClonotypeSummary:
                        color_by_label: celiac

    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None, number_of_processes: int = 1,
                 split_by_label: bool = None, label: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.split_by_label = split_by_label
        self.label_name = label

    def _set_label_name(self):
        if self.split_by_label:
            if self.label_name is None:
                self.label_name = list(self.dataset.get_label_names())[0]
        else:
            self.label_name = None

    @classmethod
    def build_object(cls, **kwargs):
        ReportUtil.update_split_by_label_kwargs(kwargs, RepertoireClonotypeSummary.__name__)

        return RepertoireClonotypeSummary(**kwargs)

    def _generate(self) -> ReportResult:
        self._set_label_name()
        PathBuilder.build(self.result_path)
        return self._safe_plot()

    def _plot(self) -> ReportResult:
        clonotypes = pd.DataFrame.from_records(sorted([self._get_clonotype_count_with_label(repertoire) for repertoire in self.dataset.get_data()],
                                                      key=lambda x: x[0]), columns=['clonotype_count', self.label_name])
        clonotypes['repertoire'] = list(range(1, self.dataset.get_example_count()+1))
        fig = px.bar(clonotypes, x='repertoire', y='clonotype_count', color=self.label_name, title='Clonotype count per repertoire',
                     color_discrete_sequence=px.colors.diverging.Tealrose)
        fig.update_layout(template="plotly_white", yaxis_title='clonotype count', xaxis_title='repertoires sorted by clonotype count')
        clonotypes.to_csv(self.result_path / 'clonotype_count_per_repertoire.csv')
        fig.write_html(str(self.result_path / 'clonotype_count_per_repertoire.html'))

        return ReportResult(name=self.name, info="Clonotype count per repertoire",
                            output_figures=[ReportOutput(self.result_path / 'clonotype_count_per_repertoire.html')],
                            output_tables=[ReportOutput(self.result_path / 'clonotype_count_per_repertoire.csv')])

    def _get_clonotype_count_with_label(self, repertoire: Repertoire) -> Tuple[int, str]:

        sequences = repertoire.data

        sequence_count = len(sequences)
        unique_sequence_count = np.unique(sequences.cdr3_aa.tolist()).shape[0]
        if sequence_count != unique_sequence_count:
            logging.warning(f"{RepertoireClonotypeSummary.__name__}: for repertoire {repertoire.identifier}, there are {sequence_count} sequences, "
                            f"but {unique_sequence_count} unique sequences.")

        return unique_sequence_count, repertoire.metadata[self.label_name] if self.split_by_label else None

    def check_prerequisites(self) -> bool:
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            logging.warning(f"{RepertoireClonotypeSummary.__name__}: report can be generated only from RepertoireDataset. Skipping this report...")
            return False