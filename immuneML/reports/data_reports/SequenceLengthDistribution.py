from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
from pandas import DataFrame

from immuneML.data_model import bnp_util, AIRRSequenceSet
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

    - split_by_label (bool): Whether to split the plots by a label. If set to true, the Dataset must either contain a
      single label, or alternatively the label of interest can be specified under 'label'. By default,
      split_by_label is False.

    - label (str): if split_by_label is set to True, a label can be specified here.

    - plot_frequencies (bool): if set to True, the plot will show the frequencies of the sequence lengths instead of the
      counts. By default, plot_frequencies is False.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_sld_report:
                    SequenceLengthDistribution:
                        sequence_type: amino_acid
                        region_type: IMGT_CDR3
                        label: label_1
                        split_by_label: True
                        plot_frequencies: True

    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_sequence_type(kwargs)
        ParameterValidator.assert_region_type(kwargs)

        return SequenceLengthDistribution(**{**kwargs, 'sequence_type': SequenceType[kwargs['sequence_type'].upper()],
                                             'region_type': RegionType[kwargs['region_type'].upper()]})

    def __init__(self, dataset: Dataset = None, batch_size: int = 1, result_path: Path = None,
                 number_of_processes: int = 1, region_type: RegionType = RegionType.IMGT_CDR3,
                 sequence_type: SequenceType = SequenceType.AMINO_ACID, name: str = None, label: str = None,
                 split_by_label: bool = False, plot_frequencies: bool = False):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.batch_size = batch_size
        self.sequence_type = sequence_type
        self.region_type = region_type
        self.label_name = label
        self.split_by_label = split_by_label
        self.plot_frequencies = plot_frequencies

    def check_prerequisites(self):
        return True

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self.label_name = self._get_label_name()

        df = self._get_sequence_lengths_df()
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
            sequence_lengths_df = self._get_sequence_lengths_df_repertoire_dataset()
        elif isinstance(self.dataset, SequenceDataset):
            sequence_lengths_df = self._get_sequence_lengths_df_sequence_dataset()
        elif isinstance(self.dataset, ReceptorDataset):
            sequence_lengths_df = self._get_sequence_lengths_df_receptor_dataset()

        if not self.split_by_label and self.label_name in sequence_lengths_df.columns:
            sequence_lengths_df.drop(columns=[self.label_name], inplace=True)

        if self.plot_frequencies:
            if self.split_by_label and 'chain' not in sequence_lengths_df.columns:
                sequence_lengths_df['frequencies'] = sequence_lengths_df.groupby(self.label_name)['counts'].transform(
                    lambda x: x / x.sum())
            elif self.split_by_label and 'chain' in sequence_lengths_df.columns:
                sequence_lengths_df['frequencies'] = sequence_lengths_df.groupby([self.label_name, 'chain'])[
                    'counts'].transform(lambda x: x / x.sum())
            else:
                sequence_lengths_df['frequencies'] = sequence_lengths_df['counts'] / sequence_lengths_df['counts'].sum()

        return sequence_lengths_df

    def _get_sequence_lengths_df_repertoire_dataset(self):
        raw_count_dict_per_class = {}
        label_name = self._get_label_name()

        if self.split_by_label:
            class_names = self.dataset.get_metadata([label_name])[label_name]
        else:
            class_names = [0] * self.dataset.get_example_count()

        for repertoire, class_name in zip(self.dataset.get_data(), class_names):
            self._count_seq_lengths(repertoire.data, raw_count_dict_per_class, class_name)

        return self._count_dict_per_class_to_df(raw_count_dict_per_class)

    def _get_sequence_lengths_df_sequence_dataset(self):
        return self._count_dict_per_class_to_df(self._count_seq_lengths(self.dataset.data, class_name=None))

    def _count_dict_per_class_to_df(self, raw_count_dict_per_class):
        result_dfs = []

        for class_name, raw_count_dict in raw_count_dict_per_class.items():
            result_df = self._count_dict_to_df(raw_count_dict)
            result_df[self.label_name] = class_name
            result_dfs.append(result_df)

        return pd.concat(result_dfs)

    def _count_seq_lengths(self, data: AIRRSequenceSet, raw_count_dict=None, class_name: str = None):
        raw_count_dict = {} if raw_count_dict is None else raw_count_dict

        for item in data.to_iter():
            sequence = getattr(item, bnp_util.get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID))

            if self.split_by_label and self.label_name is not None:
                if class_name is None:
                    cls_name = getattr(item, self.label_name)
                else:
                    cls_name = class_name
            else:
                cls_name = 0

            if cls_name not in raw_count_dict:
                raw_count_dict[cls_name] = {}

            if len(sequence) not in raw_count_dict[cls_name]:
                raw_count_dict[cls_name][len(sequence)] = 1
            else:
                raw_count_dict[cls_name][len(sequence)] += 1

        return raw_count_dict

    def _count_dict_to_df(self, count_dict):
        return pd.DataFrame({"counts": list(count_dict.values()),
                             'sequence_lengths': list(count_dict.keys())})

    def _get_dataset_chains(self):
        return next(self.dataset.get_data()).get_chains()

    def _get_sequence_lengths_df_receptor_dataset(self):
        data = self.dataset.data
        chains = list(set(data.locus.tolist()))

        dfs = []
        label_name = self._get_label_name()

        for chain in chains:
            chain_data = data[[el == chain for el in data.locus.tolist()]]

            if self.split_by_label and label_name:
                chain_df = self._count_dict_per_class_to_df(self._count_seq_lengths(chain_data, class_name=None))
                chain_df["chain"] = chain
                dfs.append(chain_df)
            else:
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

        figure = px.bar(df, x="sequence_lengths", y="frequencies" if self.plot_frequencies else "counts",
                        facet_col=self.label_name if self.label_name in df.columns else None,
                        facet_row="chain" if isinstance(self.dataset, ReceptorDataset) else None)
        figure.update_layout(template="plotly_white")
        figure.update_traces(marker_color=px.colors.diverging.Tealrose[0])

        for annotation in figure.layout.annotations:
            annotation['font'] = {'size': 16}

        PathBuilder.build(self.result_path)

        file_path = self.result_path / "sequence_length_distribution.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="Sequence length distribution plot")

    def _get_label_name(self):
        if self.split_by_label:
            if self.label_name is None:
                return list(self.dataset.get_label_names())[0]
            else:
                return self.label_name
        else:
            return None
