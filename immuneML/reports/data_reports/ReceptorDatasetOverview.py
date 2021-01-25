from pathlib import Path
from typing import Tuple, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class ReceptorDatasetOverview(DataReport):

    def __init__(self, batch_size: int, dataset: ReceptorDataset = None, result_path: Path = None, name: str = None):
        super().__init__(dataset, result_path, name)
        self.batch_size = batch_size

    @classmethod
    def build_object(cls, **kwargs):
        return ReceptorDatasetOverview(**kwargs)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        figure, tables = self._generate_sequence_length_distribution_plots()
        return ReportResult(name=self.name, output_figures=[figure], output_tables=tables)

    def _prepare_data_for_length_distribution(self):
        receptors = {}
        for receptor in self.dataset.get_data(self.batch_size):
            for chain in receptor.get_chains():
                receptor_dict = {
                    "length": len(receptor.get_chain(chain).get_sequence()),
                    "chain": chain
                }
                if chain in receptors:
                    receptors[chain].append(receptor_dict)
                else:
                    receptors[chain] = [receptor_dict]

        chains = list(receptors.keys())
        dfs = [pd.DataFrame(receptors[chain]) for chain in chains]
        return dfs, chains

    def _generate_sequence_length_distribution_plots(self) -> Tuple[ReportOutput, List[ReportOutput]]:
        dfs, chains = self._prepare_data_for_length_distribution()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dfs[0]["length"],
            histnorm='probability density',
            opacity=0.75,
            name=chains[0],
            marker={'color': px.colors.diverging.Tealrose[0]}
        ))
        fig.add_trace(go.Histogram(
            x=dfs[1]["length"],
            histnorm='probability density',
            opacity=0.75,
            name=chains[1],
            marker={'color': px.colors.diverging.Tealrose[-2]}
        ))
        fig.update_layout(title_text="Receptor sequence length distribution per chain", xaxis_title_text="receptor sequence length",
                          yaxis_title_text="frequency", bargap=0.2, bargroupgap=0.1, template="plotly_white")

        image_output, table_outputs = self._store_sequence_distribution_data(fig, dfs, chains)

        return image_output, table_outputs

    def _store_sequence_distribution_data(self, fig, dfs, chains):
        fig.write_html(str(self.result_path / "sequence_length_distribution.html"))
        image_output = ReportOutput(self.result_path / "sequence_length_distribution.html", name="sequence length distribution per chain")
        table_outputs = [ReportOutput(self.result_path / f"sequence_length_distribution_chain_{chains[index]}.csv") for index in range(len(chains))]
        for index, df in enumerate(dfs):
            df.to_csv(table_outputs[index].path, index=False)

        return image_output, table_outputs

    def check_prerequisites(self):
        if isinstance(self.dataset, ReceptorDataset):
            return True
        else:
            return False
