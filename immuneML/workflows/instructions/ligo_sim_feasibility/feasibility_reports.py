from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.simulation.generative_models.BackgroundSequences import BackgroundSequences


def report_signal_frequencies(frequencies: pd.DataFrame, path: Path) -> ReportResult:
    csv_output = ReportOutput(path / 'signal_frequencies.csv', 'signal frequencies as csv')
    frequencies.to_csv(csv_output.path, index=False)

    fig = px.bar(frequencies, x='signal', y='frequency', template='plotly_white', color_discrete_sequence=px.colors.diverging.Tealrose)
    fig_output = ReportOutput(path / 'signal_frequencies.html', 'signal frequencies bar chart')
    fig.write_html(str(fig_output.path))

    return ReportResult('signal frequencies', output_figures=[fig_output], output_tables=[csv_output])


def report_signal_cooccurrences(unique_values: np.ndarray, counts: np.ndarray, path: Path) -> ReportResult:
    df = pd.DataFrame({"signal_count": unique_values, "sequence_count": counts})
    csv_output = ReportOutput(path / 'signal_counts_per_sequence.csv', 'number of signals occurring in a single sequence')
    df.to_csv(csv_output.path, index=False)

    fig = px.bar(df, x='signal_count', y='sequence_count', template='plotly_white', color_discrete_sequence=px.colors.diverging.Tealrose)
    fig_output = ReportOutput(path / 'signal_counts_per_sequence.html', 'signal counts per sequence')
    fig.write_html(str(fig_output.path))

    return ReportResult('signal co-occurrences', output_figures=[fig_output], output_tables=[csv_output])


def make_p_gen_histogram(sequences: BackgroundSequences, p_gen_bin_count: int):
    log_p_gens = np.log10(sequences.p_gen)
    hist, p_gen_bins = np.histogram(log_p_gens, density=False, bins=np.concatenate(
        ([np.NINF], np.histogram_bin_edges(log_p_gens, p_gen_bin_count), [np.PINF])))

    return hist / len(sequences), p_gen_bins


def report_p_gen_histogram(histogram: np.ndarray, bins, path: Path) -> ReportResult:
    return ReportResult()
