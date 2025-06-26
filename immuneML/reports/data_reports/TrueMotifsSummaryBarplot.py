import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.signal_annotation import annotate_sequence_dataset
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class TrueMotifsSummaryBarplot(DataReport):
    """
    This report can be used to show how well motifs (for example, through the Simulation instruction) are learned
    across different generative models. The report shows a bar plot with the number of sequences in each dataset that
    contain the given motifs. Bars are grouped by the dataset origin (e.g., train, test, PWM, VAE, LSTM) and the
    signals provided. The report also shows the total count of sequences containing at least one signal for each dataset.

    **Specification arguments:**

    - implanted_motifs_per_signal (dict): a nested dictionary that specifies the motif seeds that were implanted in the
      given dataset. The first level of keys in this dictionary represents the different signals. In the inner
      dictionary there should be two keys: "seeds" and "gap_sizes".

      - seeds: a list of motif seeds. The seeds may contain gaps, specified by a '/' symbol.

      - gap_sizes: a list of all the possible gap sizes that were used when implanting a gapped motif seed. When no
        gapped seeds are used, this value has no effect.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_motif_report:
                    TrueMotifsSummaryBarplot:
                        implanted_motifs_per_signal:
                            signal1:
                                seeds:
                                - DEQ
                                gap_sizes:
                                - 0
                            signal2:
                                seeds:
                                - AS/G
                                gap_sizes:
                                - 2

    """
    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_keys_present(list(kwargs.keys()),
                                               ["implanted_motifs_per_signal"],
                                               "TrueMotifsSummaryBarplot",
                                               "TrueMotifsSummaryBarplot report")
        implanted_motifs_per_signal = kwargs["implanted_motifs_per_signal"]

        ParameterValidator.assert_type_and_value(implanted_motifs_per_signal, dict,
                                                 "TrueMotifsSummaryBarplot",
                                                 f"implanted_motifs_per_signal")

        for signal in implanted_motifs_per_signal.keys():
            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal], dict,
                                                     "TrueMotifsSummaryBarplot",
                                                     f"implanted_motifs_per_signal/{signal}")

            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal], dict,
                                                     "TrueMotifsSummaryBarplot",
                                                     f"implanted_motifs_per_signal/{signal}")

            ParameterValidator.assert_keys_present(implanted_motifs_per_signal[signal].keys(),
                                                   ["seeds", "gap_sizes"],
                                                   "TrueMotifsSummaryBarplot",
                                                   f"implanted_motifs_per_signal/{signal}")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal]["gap_sizes"], list,
                                                     "TrueMotifsSummaryBarplot",
                                                     f"implanted_motifs_per_signal/{signal}/gap_sizes")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal]["seeds"], list,
                                                     "TrueMotifsSummaryBarplot",
                                                     f"implanted_motifs_per_signal/{signal}/seeds")
            for gap_size in implanted_motifs_per_signal[signal]["gap_sizes"]:
                ParameterValidator.assert_type_and_value(gap_size, int, "TrueMotifsSummaryBarplot",
                                                         f"implanted_motifs_per_signal/{signal}/gap_sizes",
                                                         min_inclusive=0)
            for seed in implanted_motifs_per_signal[signal]["seeds"]:
                ParameterValidator.assert_type_and_value(seed, str, "TrueMotifsSummaryBarplot",
                                                             f"implanted_motifs_per_signal/{signal}/seeds")

        return TrueMotifsSummaryBarplot(**kwargs)

    def __init__(self, implanted_motifs_per_signal, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.implanted_motifs_per_signal = implanted_motifs_per_signal
        self.dataset = dataset
        self.result_path = result_path

    def _generate(self):
        PathBuilder.build(self.result_path)

        plot_df = self._get_plotting_data()
        plot_df.to_csv(self.result_path / 'annotated_sequences.csv', index=False)
        report_output_fig = self._plot(plot_df, "true_motifs_summary_barplot")

        return ReportResult(self.name,
                            info="This report shows how well implanted ('ground truth') motifs are recovered by "
                                 "generative models. ",
                            output_tables=[ReportOutput(self.result_path / 'annotated_sequences.csv',
                                                        'annotated sequences')],
                            output_figures=[report_output_fig])

    def _get_plotting_data(self):
        signals = self._get_implanted_signals()
        annotated_sequences = annotate_sequence_dataset(self.dataset, signals=signals,)
        return annotated_sequences

    def _plot(self, plotting_data, output_name):
        if plotting_data.empty:
            logging.warning(f"Empty data, skipping {output_name} plot...")
        else:
            filename = self.result_path / f"{output_name}.html"

            signal_names = list(self.implanted_motifs_per_signal.keys())
            signal_mask = plotting_data[signal_names].sum(axis=1) > 0
            seq_count_with_signals = plotting_data[signal_mask].groupby('data_origin').size().reset_index(
                                                                        name='seq_count_with_signal')
            sorted_data_origins = seq_count_with_signals.sort_values('seq_count_with_signal',
                                                                     ascending=False)['data_origin']

            df_grouped = plotting_data.groupby('data_origin')[signal_names].sum().reset_index()
            df_grouped['total'] = df_grouped[signal_names].sum(axis=1)
            df_grouped['data_origin'] = pd.Categorical(df_grouped['data_origin'],
                                                       categories=sorted_data_origins,
                                                       ordered=True)
            df_grouped = df_grouped.sort_values('data_origin')

            df_melted = df_grouped.drop(columns='total').melt(
                                        id_vars='data_origin',
                                        var_name='signal',
                                        value_name='total_count'
                                        )
            df_melted['signal'] = pd.Categorical(df_melted['signal'], categories=signal_names, ordered=True)
            df_melted['data_origin'] = pd.Categorical(df_melted['data_origin'], categories=sorted_data_origins,
                                                                                ordered=True)
            df_melted = df_melted.sort_values(['data_origin', 'signal'])

            seq_count_with_signals['data_origin'] = pd.Categorical(seq_count_with_signals['data_origin'],
                                                                   categories=sorted_data_origins, ordered=True)
            seq_count_with_signals = seq_count_with_signals.sort_values('data_origin')

            figure = px.bar(
                df_melted,
                x='data_origin',
                y='total_count',
                color='signal',
                color_discrete_sequence=px.colors.diverging.Tealrose,
                barmode='group',
                title='Number of sequences containing signals per dataset (Total counts on top: number of sequences '
                      'containing at least one signal)'
            )

            figure.update_layout(
                xaxis_title='Data Origin',
                yaxis_title='Sequence Count',
                bargap=0.2
            )

            for idx, row in seq_count_with_signals.iterrows():
                max_y = df_melted[df_melted['data_origin'] == row['data_origin']]['total_count'].max()
                figure.add_annotation(
                    x=row['data_origin'],
                    y=max_y + 2,
                    text="Total count = " + str(row['seq_count_with_signal']),
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    yanchor='bottom'
                )

            figure.write_html(str(filename))

            return ReportOutput(filename,
                                f"Summary of motif recovery for {self.name}",)

    def _get_implanted_signals(self):
        signals = []
        for signal in self.implanted_motifs_per_signal.keys():
            seeds = self.implanted_motifs_per_signal[signal]["seeds"]
            gap_sizes = self.implanted_motifs_per_signal[signal]["gap_sizes"]
            signals.append(Signal(signal, [SeedMotif(f'm{i+1}', seed=seed,
                                                     min_gap=min(gap_sizes),
                                                     max_gap=max(gap_sizes)) for i, seed in enumerate(seeds)]))
        return signals

    def check_prerequisites(self):
        location = "TrueMotifSummary"

        run_report = True

        if 'data_origin' not in self.dataset.labels.keys():
            logging.warning(
                f"{location}: this report can only be created for a combined dataset exported with TrainGenModel "
                f"instruction. Report {self.name} will not be created.")
            run_report = False

        return run_report

