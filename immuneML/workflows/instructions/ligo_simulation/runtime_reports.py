from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml


def make_p_gen_histogram_plot(hist_data: np.ndarray, hist_bin_edges: np.ndarray, path: Path, p_gen_text: str):

    with (path / 'histogram_raw_values.yaml').open('w') as file:
        yaml.dump({'histogram_data': hist_data.tolist(), 'histogram_bin_edges': hist_bin_edges.tolist()}, file)

    fig = go.Figure(data=[go.Bar(x=hist_bin_edges, y=hist_data, marker={'colorscale': 'Tealrose'})])
    fig.update_layout(template='plotly_white',
                      xaxis_title_text=f'log10 generation probability histogram<br>(sequences with log10 Pgen '
                                       f'outside this range have probability of {p_gen_text})')
    fig.write_html(str(path / 'p_gen_histogram.html'))
