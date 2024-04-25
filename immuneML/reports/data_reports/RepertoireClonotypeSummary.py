import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
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

    def __init__(self, dataset: Dataset = None, result_path: Path = None, name: str = None, number_of_processes: int = 1, color_by_label: str = None):
        super().__init__(dataset, result_path, name, number_of_processes)
        self.color_by_label = color_by_label

    @classmethod
    def build_object(cls, **kwargs):
        if "color_by_label" in kwargs and kwargs['color_by_label'] is not None:
            ParameterValidator.assert_type_and_value(kwargs['color_by_label'], str, RepertoireClonotypeSummary.__name__, 'color_by_label')

        return RepertoireClonotypeSummary(**kwargs)

    def _generate(self) -> ReportResult:
        assert isinstance(self.dataset, RepertoireDataset), \
            f"{RepertoireClonotypeSummary.__name__}: expected repertoire dataset, but got {type(self.dataset)}."

        PathBuilder.build(self.result_path)
        return self._safe_plot()

    def _plot(self) -> ReportResult:
        clonotypes = pd.DataFrame.from_records(sorted([self._get_clonotype_count_with_label(repertoire) for repertoire in self.dataset.get_data()],
                                                      key=lambda x: x[0]), columns=['clonotype_count', self.color_by_label])
        clonotypes['repertoire'] = list(range(1, self.dataset.get_example_count()+1))
        fig = px.bar(clonotypes, x='repertoire', y='clonotype_count', color=self.color_by_label, title='Clonotype count per repertoire',
                     color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig.update_layout(template="plotly_white", yaxis_title='clonotype count', xaxis_title='repertoires sorted by clonotype count')
        clonotypes.to_csv(self.result_path / 'clonotype_count_per_repertoire.csv')
        fig.write_html(str(self.result_path / 'clonotype_count_per_repertoire.html'))

        return ReportResult(name=self.name, info="Clonotype count per repertoire",
                            output_figures=[ReportOutput(self.result_path / 'clonotype_count_per_repertoire.html')],
                            output_tables=[ReportOutput(self.result_path / 'clonotype_count_per_repertoire.csv')])

    def _get_clonotype_count_with_label(self, repertoire: Repertoire) -> Tuple[int, str]:

        sequences = repertoire.get_attribute('sequences')
        if sequences is None:
            sequences = repertoire.get_sequence_aas()

        sequence_count = sequences.shape[0]
        unique_sequence_count = np.unique(sequences.tolist()).shape[0]
        if sequence_count != unique_sequence_count:
            logging.warning(f"{RepertoireClonotypeSummary.__name__}: for repertoire {repertoire.identifier}, there are {sequence_count} sequences, "
                            f"but {unique_sequence_count} unique sequences.")

        return unique_sequence_count, repertoire.metadata[self.color_by_label] if self.color_by_label else None
