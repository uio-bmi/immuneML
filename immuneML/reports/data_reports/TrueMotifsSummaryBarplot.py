import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from immuneML.data_model import bnp_util
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.PlotlyUtil import PlotlyUtil
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.util.signal_annotation import annotate_sequence_dataset
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.CompAIRRParams import CompAIRRParams
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class TrueMotifsSummaryBarplot(DataReport):
    """
    This report can be used to show how well motifs (for example, motifs introduced using the Simulation instruction)
    are learned across different generative models. The report shows a bar plot with the proportion of sequences in each
    dataset that contain the given motifs. Bars are grouped by the dataset origin (e.g., train, PWM, VAE, LSTM)
    and the signals provided. The report also shows how many of the sequences are memorized (seen in train data) and
    how many are novel (not seen in train data).

    **Specification arguments:**

    - region_type (str): which part of the sequence to check; e.g., IMGT_CDR3

    - max_edit_distance (int): how far from a training sequence a generated sequence may be and still count as
      memorized, between 0 and 2. With the default of 0, only generated sequences that are identical to a training
      sequence count as memorized, so a single amino acid substitution makes a copied sequence count as novel. With
      1 or 2, the bars are split further into 'within 1 edit' and 'within 2 edits', which shows how much of what
      looks novel is a near-copy. The differences are substitutions (CompAIRR is run with -d and without -i, since
      CompAIRR only allows indels when d=1), so sequences of different lengths never match.

    - compairr_path (str): path to the CompAIRR executable, used only when max_edit_distance is above 0. If not
      given, it is assumed that CompAIRR can be called as 'compairr' on the command line, or that it is at
      /usr/local/bin/compairr.

    - threads (int): the number of threads CompAIRR may use; this only affects speed. By default, 8.

    - show_test_set_baseline (bool): whether to score the held-out sequences (data_origin 'original_test') against the
      training set in the same way as the generated ones and show them alongside. They are independent draws from the
      same source as the training sequences, so the rate they reach is what 'not memorizing' looks like at this sample
      size, and a model is only memorizing insofar as it exceeds that. By default, this is False, and the held-out
      sequences are left out of the plot.

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
                        region_type: IMGT_CDR3
                        implanted_motifs_per_signal:
                            my_signal1:
                                seeds:
                                - DEQ
                                gap_sizes:
                                - 0
                            my_signal2:
                                seeds:
                                - AS/G
                                gap_sizes:
                                - 2
                        max_edit_distance: 2

    """
    @classmethod
    def build_object(cls, **kwargs):
        location = TrueMotifsSummaryBarplot.__name__
        ParameterValidator.assert_keys_present(list(kwargs.keys()),
                                               ["implanted_motifs_per_signal", "region_type"],
                                               location, "TrueMotifsSummaryBarplot report")
        ParameterValidator.assert_region_type(kwargs, location)
        implanted_motifs_per_signal = kwargs["implanted_motifs_per_signal"]

        ParameterValidator.assert_type_and_value(implanted_motifs_per_signal, dict,
                                                 location, f"implanted_motifs_per_signal")

        for signal in implanted_motifs_per_signal.keys():
            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal], dict, location,
                                                     f"implanted_motifs_per_signal/{signal}")

            ParameterValidator.assert_keys_present(implanted_motifs_per_signal[signal].keys(),
                                                   ["seeds", "gap_sizes"], location,
                                                   f"implanted_motifs_per_signal/{signal}")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal]["gap_sizes"], list, location,
                                                     f"implanted_motifs_per_signal/{signal}/gap_sizes")
            ParameterValidator.assert_type_and_value(implanted_motifs_per_signal[signal]["seeds"], list, location,
                                                     f"implanted_motifs_per_signal/{signal}/seeds")
            for gap_size in implanted_motifs_per_signal[signal]["gap_sizes"]:
                ParameterValidator.assert_type_and_value(gap_size, int, location,
                                                         f"implanted_motifs_per_signal/{signal}/gap_sizes",
                                                         min_inclusive=0)
            for seed in implanted_motifs_per_signal[signal]["seeds"]:
                ParameterValidator.assert_type_and_value(seed, str, location,
                                                         f"implanted_motifs_per_signal/{signal}/seeds")

        max_edit_distance = kwargs.get("max_edit_distance", 0)
        ParameterValidator.assert_type_and_value(max_edit_distance, int, location, "max_edit_distance",
                                                 min_inclusive=0, max_inclusive=2)
        if max_edit_distance > 0:
            CompAIRRHelper.check_compairr_path(kwargs.get("compairr_path", None))

        ParameterValidator.assert_type_and_value(kwargs.get("show_test_set_baseline", False), bool, location,
                                                 "show_test_set_baseline")

        return TrueMotifsSummaryBarplot(**{**kwargs, 'region_type': RegionType[kwargs['region_type'].upper()]})

    def __init__(self, implanted_motifs_per_signal, dataset: Dataset = None, result_path: Path = None, name: str = None,
                 number_of_processes: int = 1, region_type: RegionType = RegionType.IMGT_CDR3,
                 max_edit_distance: int = 0, compairr_path: str = None, threads: int = 8,
                 show_test_set_baseline: bool = False):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.implanted_motifs_per_signal = implanted_motifs_per_signal
        self.dataset = dataset
        self.result_path = result_path
        self.region_type = region_type
        self.max_edit_distance = max_edit_distance
        self.compairr_path = compairr_path
        self.threads = threads
        self.show_test_set_baseline = show_test_set_baseline

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
        annotated_sequences = annotate_sequence_dataset(self.dataset, signals=signals, region_type=self.region_type)
        annotated_sequences = self._add_novelty_memorization_label(annotated_sequences)
        return annotated_sequences

    def _add_novelty_memorization_label(self, plotting_data):
        sequence_column = bnp_util.get_sequence_field_name(self.region_type, SequenceType.AMINO_ACID)
        train_set = set(plotting_data.loc[plotting_data['data_origin'] == 'original_train', sequence_column])

        # created up front as strings: filling it in row by row would otherwise start it off as floats
        plotting_data['novelty_memorization'] = pd.Series(dtype=str, index=plotting_data.index)

        # the held-out sequences are scored like the generated ones when they are shown as the baseline: they are
        # independent draws from the same source, so the rate they reach is what not memorizing looks like
        originals = ['original_train'] if self.show_test_set_baseline else ['original_train', 'original_test']

        for index, row in plotting_data.iterrows():
            if row['data_origin'] in originals:
                plotting_data.at[index, 'novelty_memorization'] = 'original'
            else:
                seq = row[sequence_column]
                if seq in train_set:
                    plotting_data.at[index, 'novelty_memorization'] = 'memorized'
                else:
                    plotting_data.at[index, 'novelty_memorization'] = 'novel'

        if self.max_edit_distance > 0:
            plotting_data = self._label_near_copies(plotting_data, sequence_column)

        return plotting_data

    def _label_near_copies(self, plotting_data, sequence_column):
        """Relabel the sequences counted as novel that are within max_edit_distance of a training sequence.

        CompAIRR is run once per distance, from 1 up to max_edit_distance, over the sequences still labelled novel,
        so each sequence ends up labelled with the smallest distance at which it matched.
        """
        train_sequences = plotting_data.loc[plotting_data['data_origin'] == 'original_train', sequence_column]
        if train_sequences.empty:
            logging.warning(f"{self.__class__.__name__}: no sequences with data_origin 'original_train', so the "
                            f"distance to the training set cannot be computed. Keeping the exact-match labels.")
            return plotting_data

        work_path = PathBuilder.build(self.result_path / "compairr_data")
        reference_path = work_path / "training_sequences.tsv"
        self._write_compairr_file(train_sequences, reference_path, "original_train")

        for distance in range(1, self.max_edit_distance + 1):
            still_novel = plotting_data[plotting_data['novelty_memorization'] == 'novel']
            if still_novel.empty:
                break

            query_path = work_path / f"generated_sequences_d{distance}.tsv"
            self._write_compairr_file(still_novel[sequence_column], query_path, "generated")

            matched = self._run_compairr(query_path, reference_path, work_path, distance,
                                         [str(index) for index in still_novel.index])
            label = f"within {distance} edit" if distance == 1 else f"within {distance} edits"
            plotting_data.loc[still_novel.index[matched], 'novelty_memorization'] = label

        return plotting_data

    @staticmethod
    def _write_compairr_file(sequences, filename, repertoire_id):
        """CompAIRR input with genes and counts ignored, so only the amino acid sequence is compared."""
        pd.DataFrame({"junction_aa": sequences.values,
                      "repertoire_id": repertoire_id,
                      "sequence_id": [str(index) for index in sequences.index]}).to_csv(filename, sep="\t",
                                                                                        index=False)

    def _run_compairr(self, query_path, reference_path, work_path, distance, query_ids):
        import subprocess

        compairr_params = CompAIRRParams(compairr_path=CompAIRRHelper.determine_compairr_path(self.compairr_path),
                                         keep_compairr_input=True, differences=distance, indels=False,
                                         ignore_counts=True, ignore_genes=True, threads=self.threads,
                                         output_filename=f"compairr_out_d{distance}.txt",
                                         log_filename=f"compairr_log_d{distance}.txt",
                                         output_pairs=False, pairs_filename=None,
                                         do_repertoire_overlap=False, do_sequence_matching=True)

        args = CompAIRRHelper.get_cmd_args(compairr_params, [query_path, reference_path], work_path)
        logging.info(f"{self.__class__.__name__}: running CompAIRR with the following arguments: {' '.join(args)}")
        result = CompAIRRHelper.process_compairr_output_file(subprocess.run(args, capture_output=True, text=True),
                                                             compairr_params, work_path)

        # CompAIRR does not have to return the sequences in the order they were given in
        result.index = result.index.astype(str)
        assert set(result.index) == set(query_ids), \
            (f"{self.__class__.__name__}: CompAIRR returned {len(set(result.index))} sequence identifiers for "
             f"{len(set(query_ids))} query sequences. See the log file in {work_path} for more information.")

        return (result.reindex(query_ids)["original_train"] > 0).values

    @staticmethod
    def _get_sequence_counts(plotting_data, signal_names):
        signal_mask = plotting_data[signal_names].sum(axis=1) > 0
        seq_count_with_signals = plotting_data[signal_mask].groupby('data_origin').size().reset_index(
                                                                        name='seq_count_with_signal')

        seq_count_total = plotting_data.groupby('data_origin').size().reset_index(name='total_sequences')
        seq_counts_df = seq_count_with_signals.merge(seq_count_total, on='data_origin', how='left')
        seq_counts_df['signal_specific_percent'] = seq_counts_df['seq_count_with_signal'] / seq_counts_df['total_sequences'] * 100
        seq_counts_df['label'] = seq_counts_df.apply(
            lambda row: f"{row['data_origin']}<br>Signal-specific sequences: {row['signal_specific_percent']:.2f}%", axis=1)

        return seq_counts_df

    @staticmethod
    def _prepare_and_sort_plotting_data(plotting_data, signal_names, sorted_data_origins, seq_counts_df):
        df_grouped = plotting_data.groupby(['data_origin', 'novelty_memorization'])[signal_names].sum().reset_index()
        df_grouped['total_count'] = df_grouped[signal_names].sum(axis=1)

        df_melted = df_grouped.drop(columns='total_count').melt(id_vars=['data_origin', 'novelty_memorization'],
                                                                var_name='signal',
                                                                value_name='count')
        df_melted['data_origin'] = pd.Categorical(df_melted['data_origin'],
                                                  categories=sorted_data_origins,
                                                  ordered=True)

        df_melted = df_melted.sort_values(['data_origin', 'novelty_memorization', 'signal'])
        df_melted = df_melted.merge(seq_counts_df, on='data_origin', how='left')
        df_melted['frequency'] = df_melted.groupby(['data_origin', 'signal'])['count'].transform(
            lambda x: x / df_melted['total_sequences'])

        return df_melted

    def _plot(self, plotting_data, output_name):
        if plotting_data.empty:
            logging.warning(f"Empty data, skipping {output_name} plot...")
        else:
            filename = self.result_path / f"{output_name}.html"

            if not self.show_test_set_baseline:
                plotting_data = plotting_data[plotting_data['data_origin'] != 'original_test']
            signal_names = list(self.implanted_motifs_per_signal.keys())

            seq_counts_df = self._get_sequence_counts(plotting_data, signal_names)
            # the original sequences come first, the held-out baseline next to them, then the models
            seq_counts_df['sort_key'] = seq_counts_df['data_origin'].apply(
                lambda x: (0, 0) if x == 'original_train' else (0, 1) if x == 'original_test' else (
                    1, -seq_counts_df.loc[seq_counts_df['data_origin'] == x, 'signal_specific_percent'].iloc[0]))
            sorted_data_origins = seq_counts_df.sort_values('sort_key')['data_origin'].tolist()

            df_melted = self._prepare_and_sort_plotting_data(plotting_data, signal_names, sorted_data_origins,
                                                             seq_counts_df)
            df_melted['data_origin_novelty'] = df_melted['data_origin'] + ' - ' + df_melted['novelty_memorization']

            figure = px.bar(
                df_melted,
                x='signal',
                y='frequency',
                color='data_origin_novelty',
                facet_col='label',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                barmode='stack',
                title='Percentage of sequences containing signals across different generated datasets',
            )

            # make bars grouped by data_origin and stacked by novelty
            # figure.update_traces(offsetgroup=df_melted["data_origin"],  # group by data_origin
            #                      opacity=1.0)

            # now reduce opacity for each novelty_memorization level
            color_map = {data_origin: px.colors.qualitative.Vivid[i]
                         for i, data_origin in enumerate(plotting_data['data_origin'].unique())}
            # the near-copy levels sit between memorized and novel, so the bars shade from one to the other
            opacity_map = {"original": 1.0, "memorized": 1., "within 1 edit": 0.8, "within 2 edits": 0.65,
                           "novel": 0.5}
            for trace in figure.data:
                data_origin, novelty = trace.name.split(' - ')
                trace.opacity = opacity_map.get(novelty, 0.5)
                trace.marker.color = color_map[data_origin]

            figure.for_each_annotation(lambda a: a.update(text=a.text.replace("label=", "")))

            figure.update_layout(
                xaxis_title='Signal',
                yaxis_title='Frequency',
                bargap=0.2, template='plotly_white'
            )

            filename = PlotlyUtil.write_image_to_file(figure, filename)

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
        location = "TrueMotifSummaryBarplot"

        run_report = True

        if 'data_origin' not in self.dataset.labels.keys():
            logging.warning(
                f"{location}: this report can only be created for a combined dataset exported with TrainGenModel "
                f"instruction. Report {self.name} will not be created.")
            run_report = False

        return run_report

