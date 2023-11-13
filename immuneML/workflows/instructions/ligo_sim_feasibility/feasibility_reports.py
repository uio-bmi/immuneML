from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult


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
    fig.update_layout(xaxis_title_text='signal count', yaxis_title_text='sequence count', xaxis={"tickmode": 'linear', "tick0": 0, "dtick": 1})
    fig_output = ReportOutput(path / 'signal_counts_per_sequence.html', 'signal counts per sequence')
    fig.write_html(str(fig_output.path))

    return ReportResult('signal co-occurrences', output_figures=[fig_output], output_tables=[csv_output])


def report_signal_joint_probs(signal_matrix: np.ndarray, signal_names: list, path: Path) -> ReportResult:
    joint_probs = np.zeros(shape=(len(signal_names), len(signal_names)))
    for outer_index, outer_signal in enumerate(signal_names):
        for inner_index, inner_signal in enumerate(signal_names):
            if inner_index == outer_index:
                joint_probs[inner_index, outer_index] = 1.
            elif inner_index < outer_index:
                joint_probs[inner_index, outer_index] = joint_probs[outer_index, inner_index]
            else:
                joint_probs[inner_index, outer_index] = np.logical_and(signal_matrix[:, inner_index], signal_matrix[:, outer_index]).sum() / signal_matrix.shape[0]

    table_output = ReportOutput(path / 'joint_probabilities_of_signals_co-occurring.csv', "joint probabilities of signals co-occurring")
    df = pd.DataFrame(joint_probs, index=signal_names, columns=signal_names)
    df.to_csv(str(table_output.path))

    fig = px.imshow(df, text_auto=True, color_continuous_scale='Aggrnyl')
    fig_output = ReportOutput(path / 'joint_probabilities_of_signals_co-occurring.html', "joint probabilities of signals co-occurring")
    fig.write_html(str(fig_output.path))

    return ReportResult('joint probabilities of signals co-occurring', output_figures=[fig_output], output_tables=[table_output])


def report_signal_cond_probs(signal_matrix: np.ndarray, signal_names: list, path: Path) -> ReportResult:
    cond_probs = np.zeros(shape=(len(signal_names), len(signal_names)))

    for outer_index, outer_signal in enumerate(signal_names):
        for inner_index, inner_signal in enumerate(signal_names):
            if inner_index == outer_index:
                cond_probs[outer_index, inner_index] = 1.
            else:
                cond_probs[outer_index, inner_index] = np.logical_and(signal_matrix[:, inner_index], signal_matrix[:, outer_index]).sum() / signal_matrix[:, inner_index].sum()

    description = "conditional probabilities of one signal given another signal"

    table_output = ReportOutput(path / 'cond_probabilities_of_signals_co-occurring.csv', description)
    df = pd.DataFrame(cond_probs, index=signal_names, columns=signal_names)
    df.to_csv(str(table_output.path))

    hovertext = [[f'P({signal_names[outer_ind]} | {signal_names[inner_ind]}) = ' \
                  f'{round(df.values[outer_ind, inner_ind], 3) if df.values[outer_ind, inner_ind] != np.nan else np.nan}'
                  for inner_ind in range(len(signal_names))] for outer_ind in range(len(signal_names))]

    fig = px.imshow(df, color_continuous_scale='Aggrnyl')
    fig.update_traces(text=df.values.round(3), texttemplate="%{text}", customdata=hovertext,
                      hovertemplate="%{customdata}<extra></extra>")
    fig.update_layout(template='plotly_white', xaxis_title='', yaxis_title='')
    fig_output = ReportOutput(path / 'cond_probabilities_of_signals_co-occurring.html', description)
    fig.write_html(str(fig_output.path))

    return ReportResult(description, output_figures=[fig_output], output_tables=[table_output])


def report_p_gen_histogram(sequences: BackgroundSequences, p_gen_bin_count: int, path: Path) -> ReportResult:
    log_p_gens = np.log10(sequences.p_gen)
    signal_info = ["_".join(s for index, s in enumerate(sequences.get_signal_names()) if el[index] == 1) for el in sequences.get_signal_matrix()]
    signal_info = [s if s != "" else "no signal" for s in signal_info]
    p_gen_df = pd.DataFrame({'log_p_gen': log_p_gens, "signal": signal_info})
    csv_output = ReportOutput(path / 'log10_p_gens.csv', 'generation probabilities on log10 scale')
    p_gen_df.to_csv(csv_output.path, index=False)

    fig_all = px.histogram(p_gen_df, x='log_p_gen', nbins=p_gen_bin_count + 1, template='plotly_white',
                           color_discrete_sequence=px.colors.diverging.Tealrose, histnorm='probability density')
    fig_all.update_layout(xaxis_title_text="logarithm of generation probability")
    fig_output_all = ReportOutput(path / 'log10_p_gens.html', 'generation probabilities on log10 scale')
    fig_all.write_html(str(fig_output_all.path))

    fig_signal = px.histogram(p_gen_df, x='log_p_gen', nbins=p_gen_bin_count + 1, template='plotly_white', color='signal', opacity=0.7,
                              color_discrete_sequence=px.colors.diverging.Tealrose, histnorm='probability density')
    fig_signal.update_layout(xaxis_title_text="logarithm of generation probability")
    fig_output_signal = ReportOutput(path / 'log10_p_gens_per_signal.html', 'generation probabilities on log10 scale per signal')
    fig_signal.write_html(str(fig_output_signal.path))

    return ReportResult('generation probabilities on log10 scale', output_figures=[fig_output_all, fig_output_signal], output_tables=[csv_output])


def report_seq_len_dist(sequences: BackgroundSequences, sequence_type: SequenceType, path: Path) -> ReportResult:
    lengths = sequences.get_sequence(sequence_type).lengths

    len_df = pd.DataFrame({"length": lengths})
    csv_output = ReportOutput(path / 'sequence_lengths.csv', 'sequence lengths')
    len_df.to_csv(csv_output.path, index=False)

    fig = px.histogram(len_df, x='length', template='plotly_white', color_discrete_sequence=px.colors.diverging.Tealrose,
                       histnorm='probability density')
    fig_output = ReportOutput(path / 'sequence_length_hist.html', 'sequence length histogram')
    fig.write_html(str(fig_output.path))

    return ReportResult('sequence length distribution', output_tables=[csv_output], output_figures=[fig_output])
