
import warnings
from scipy.stats import lognorm
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper


class MotifPerformancePlotHelper():

    @staticmethod
    def get_plotting_data(training_encoded_data, test_encoded_data, highlight_motifs_path=None, highlight_motifs_name="highlight"):
        training_feature_annotations = MotifPerformancePlotHelper._get_annotated_feature_annotations(training_encoded_data, highlight_motifs_path, highlight_motifs_name)
        test_feature_annotations = MotifPerformancePlotHelper._get_annotated_feature_annotations(test_encoded_data, highlight_motifs_path, highlight_motifs_name)

        training_feature_annotations["training_TP"] = training_feature_annotations["TP"]
        test_feature_annotations = MotifPerformancePlotHelper.merge_train_test_feature_annotations(training_feature_annotations, test_feature_annotations)

        return training_feature_annotations, test_feature_annotations

    @staticmethod
    def _get_annotated_feature_annotations(encoded_data, highlight_motifs_path, highlight_motifs_name):
        feature_annotations = encoded_data.feature_annotations.copy()
        MotifPerformancePlotHelper._annotate_confusion_matrix(feature_annotations)
        MotifPerformancePlotHelper._annotate_highlight(feature_annotations, highlight_motifs_path, highlight_motifs_name)

        return feature_annotations

    @staticmethod
    def _annotate_confusion_matrix(feature_annotations):
        feature_annotations["precision"] = feature_annotations.apply(
            lambda row: 0 if row["TP"] == 0 else row["TP"] / (row["TP"] + row["FP"]), axis="columns")

        feature_annotations["recall"] = feature_annotations.apply(
            lambda row: 0 if row["TP"] == 0 else row["TP"] / (row["TP"] + row["FN"]), axis="columns")

    @staticmethod
    def _annotate_highlight(feature_annotations, highlight_motifs_path, highlight_motifs_name):
        feature_annotations["highlight"] = MotifPerformancePlotHelper._get_highlight(feature_annotations, highlight_motifs_path, highlight_motifs_name)

    @staticmethod
    def _get_highlight(feature_annotations, highlight_motifs_path, highlight_motifs_name):
        if highlight_motifs_path is not None:
            highlight_motifs = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
                                for indices, amino_acids in PositionalMotifHelper.read_motifs_from_file(highlight_motifs_path)]

            return [highlight_motifs_name if motif in highlight_motifs else "Motif" for motif in
                    feature_annotations["feature_names"]]
        else:
            return ["Motif"] * len(feature_annotations)

    @staticmethod
    def merge_train_test_feature_annotations(training_feature_annotations, test_feature_annotations):
        training_info_to_merge = training_feature_annotations[["feature_names", "training_TP"]].copy()
        test_info_to_merge = test_feature_annotations.copy()

        merged_train_test_info = training_info_to_merge.merge(test_info_to_merge)

        return merged_train_test_info

    @staticmethod
    def get_combined_precision(plotting_data, min_points_in_window, smoothing_constant1, smoothing_constant2):
        group_by_tp = plotting_data.groupby("training_TP")

        combined_precision = group_by_tp["TP"].sum() / (group_by_tp["TP"].sum() + group_by_tp["FP"].sum())

        df = pd.DataFrame({"training_TP": list(combined_precision.index),
                           "combined_precision": list(combined_precision)})

        df["smooth_combined_precision"] = MotifPerformancePlotHelper._smooth_combined_precision(list(combined_precision.index),
                                                                                                    list(combined_precision),
                                                                                                    list(group_by_tp["TP"].count()),
                                                                                                    min_points_in_window,
                                                                                                    smoothing_constant1,
                                                                                                    smoothing_constant2)

        return df

    @staticmethod
    def _smooth_combined_precision(x, y, weights, min_points_in_window, smoothing_constant1, smoothing_constant2):
        smoothed_y = []

        for i in range(len(x)):
            scale = MotifPerformancePlotHelper._get_lognorm_scale(x, i, weights, min_points_in_window, smoothing_constant1, smoothing_constant2)

            lognorm_for_this_x = lognorm.pdf(x, s=0.1, loc=x[i] - scale, scale=scale)

            smoothed_y.append(sum(lognorm_for_this_x * y) / sum(lognorm_for_this_x))

        return smoothed_y

    @staticmethod
    def _get_lognorm_scale(x, i, weights, min_points_in_window, smoothing_constant1, smoothing_constant2):
        window_size = MotifPerformancePlotHelper._determine_window_size(x, i, weights, min_points_in_window)
        return window_size * smoothing_constant1 + smoothing_constant2

    @staticmethod
    def _determine_window_size(x, i, weights, min_points_in_window):
        x_rng = 0
        n_data_points = weights[i]

        if sum(weights) < min_points_in_window:
            warnings.warn(f"{MotifPerformancePlotHelper.__name__}: min_points_in_window ({min_points_in_window}) is smaller than the total number of points in the plot ({sum(weights)}). Setting min_points_in_window to {sum(weights)} instead...")
            min_points_in_window = sum(weights)
        else:
            min_points_in_window = min_points_in_window

        while n_data_points < min_points_in_window:
            x_rng += 1

            to_select = [j for j in range(len(x)) if (x[i] - x_rng) <= x[j] <= (x[i] + x_rng)]
            lower_index = min(to_select)
            upper_index = max(to_select)

            n_data_points = sum(weights[lower_index:upper_index + 1])

        return x_rng

    @staticmethod
    def get_precision_per_tp_fig(plotting_data, combined_precision, dataset_type, training_set_name,
                                 tp_cutoff=None,
                                 highlight_motifs_name="highlight"):
        fig = px.strip(plotting_data,
                       y="precision", x="training_TP", hover_data=["feature_names"],
                       range_y=[0, 1.01], color_discrete_sequence=["#74C4C4"],
                       # color="highlight",
                       # color_discrete_map={"Motif": "#74C4C4",
                       #                     self.highlight_motifs_name: px.colors.qualitative.Pastel[1]},
                       stripmode='overlay', log_x=True,
                       labels={
                           "precision": f"Precision ({dataset_type})",
                           "feature_names": "Motif",
                           "training_TP": f"True positive predictions ({training_set_name})"
                       })

        # add combined precision
        fig.add_trace(go.Scatter(x=combined_precision["training_TP"], y=combined_precision["combined_precision"],
                                 mode='markers+lines', name="Combined precision",
                                 marker=dict(symbol="diamond", color=px.colors.diverging.Tealrose[0])),
                      secondary_y=False)

        # add smoothed combined precision
        if "smooth_combined_precision" in combined_precision:
            fig.add_trace(go.Scatter(x=combined_precision["training_TP"], y=combined_precision["smooth_combined_precision"],
                                     marker=dict(color=px.colors.diverging.Tealrose[-1]),
                                     name="Combined precision, smoothed",
                                     mode="lines", line_shape='spline', line={'smoothing': 1.3}),
                          secondary_y=False, )

        # add highlighted motifs
        plotting_data_highlight = plotting_data[plotting_data["highlight"] != "Motif"]
        if len(plotting_data_highlight) > 0:
            fig.add_trace(go.Scatter(x=plotting_data_highlight["training_TP"], y=plotting_data_highlight["precision"],
                                     mode='markers', name=highlight_motifs_name,
                                     marker=dict(symbol="circle", color="#F5C144")),
                          secondary_y=False)

        # add vertical TP cutoff line
        if tp_cutoff is not None:
            fig.add_vline(x=tp_cutoff, line_dash="dash")

        fig.update_layout(xaxis=dict(dtick=1), showlegend=True)

        return fig