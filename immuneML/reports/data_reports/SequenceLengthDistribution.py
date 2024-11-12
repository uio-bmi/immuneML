from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
from pandas import DataFrame

from immuneML.data_model import bnp_util
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceLengthDistribution(DataReport):
    """
    Generates a histogram of the lengths of the sequences in a dataset.


    **Specification arguments:**

    - sequence_type (str): whether to check the length of amino acid or nucleotide sequences; default value is 'amino_acid'

    - region_type (str): which part of the sequence to examine; e.g., IMGT_CDR3

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_sld_report:
                    SequenceLengthDistribution:
                        sequence_type: amino_acid
                        region_type: IMGT_CDR3

    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_sequence_type(kwargs)
        ParameterValidator.assert_region_type(kwargs)

        return SequenceLengthDistribution(**{**kwargs, 'sequence_type': SequenceType[kwargs['sequence_type'].upper()],
                                             'region_type': RegionType[kwargs['region_type'].upper()]})

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None,
                 number_of_processes: int = 1, region_type: RegionType = RegionType.IMGT_CDR3,
                 sequence_type: SequenceType = SequenceType.AMINO_ACID, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.batch_size = batch_size
        self.sequence_type = sequence_type
        self.region_type = region_type

    def check_prerequisites(self):
        return True

    def _generate(self) -> ReportResult:
        df = self._get_sequence_lengths_df()
        PathBuilder.build(self.result_path)

        df.to_csv(self.result_path / 'sequence_length_distribution.csv', index=False)

        report_output_fig = self._safe_plot(df=df, output_written=False)
        output_figures = None if report_output_fig is None else [report_output_fig]
        return ReportResult(name=self.name,
                            info="A histogram of the lengths of the sequences in a dataset.",
                            output_figures=output_figures,
                            output_tables=[ReportOutput(self.result_path / 'sequence_length_distribution.csv',
                                                        'lengths of sequences in the dataset')])

    def _get_sequence_lengths_df(self) -> DataFrame:
        if isinstance(self.dataset, RepertoireDataset):
            return self._get_sequence_lengths_df_repertoire_dataset()
        elif isinstance(self.dataset, SequenceDataset):
            return self._get_sequence_lengths_df_sequence_dataset()
        elif isinstance(self.dataset, ReceptorDataset):
            return self._get_sequence_lengths_df_receptor_dataset()

    def _get_sequence_lengths_df_repertoire_dataset(self):
        sequence_lengths = Counter()

        for repertoire in self.dataset.get_data(self.batch_size):
            seq_lengths = self._count_in_repertoire(repertoire)
            sequence_lengths += seq_lengths

        return pd.DataFrame({"counts": list(sequence_lengths.values()),
                             'sequence_lengths': list(sequence_lengths.keys())})

    def _get_sequence_lengths_df_sequence_dataset(self):
        sequence_lengths = Counter(getattr(self.dataset.data,
                                           bnp_util.get_sequence_field_name(self.region_type,
                                                                            self.sequence_type)).lengths.tolist())

        return pd.DataFrame({"counts": list(sequence_lengths.values()),
                             'sequence_lengths': list(sequence_lengths.keys())})

    def _get_dataset_chains(self):
        return next(self.dataset.get_data()).get_chains()

    def _get_sequence_lengths_df_receptor_dataset(self):
        data = self.dataset.data
        chains = list(set(data.locus.tolist()))

        dfs = []

        for chain in chains:
            chain_data = data[[el == chain for el in data.locus.tolist()]]
            chain_counter = Counter(getattr(chain_data,
                                            bnp_util.get_sequence_field_name(self.region_type,
                                                                             self.sequence_type)).lengths.tolist())
            dfs.append(pd.DataFrame({'counts': list(chain_counter.values()),
                                     'sequence_lengths': list(chain_counter.keys()),
                                     'chain': chain}))

        return pd.concat(dfs)

    def _count_in_repertoire(self, repertoire: Repertoire) -> Counter:
        return Counter(getattr(repertoire.data,
                                bnp_util.get_sequence_field_name(self.region_type,
                                                                 self.sequence_type)).lengths.tolist())

    def _plot(self, df: pd.DataFrame) -> ReportOutput:
        figure = px.bar(df, x="sequence_lengths", y="counts",
                        facet_col="chain" if isinstance(self.dataset, ReceptorDataset) else None)
        figure.update_layout(xaxis=dict(tickmode='array', tickvals=df["sequence_lengths"]),
                             yaxis=dict(tickmode='array', tickvals=df["counts"]),
                             template="plotly_white")
        figure.update_traces(marker_color=px.colors.diverging.Tealrose[0])
        PathBuilder.build(self.result_path)

        file_path = self.result_path / "sequence_length_distribution.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Sequence length distribution plot")
