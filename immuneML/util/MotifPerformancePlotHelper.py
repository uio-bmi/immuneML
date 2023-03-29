
import warnings
from scipy.stats import lognorm
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.reports.ReportOutput import ReportOutput


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
            # highlight_motifs = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
            #                     for indices, amino_acids in PositionalMotifHelper.read_motifs_from_file(highlight_motifs_path)]

            highlight_motifs = PositionalMotifHelper.read_motifs_from_file(highlight_motifs_path)
            motifs = [PositionalMotifHelper.string_to_motif(motif, value_sep="&", motif_sep="-") for motif in feature_annotations["feature_names"]]

            return [highlight_motifs_name if MotifPerformancePlotHelper._is_highlight_motif(motif, highlight_motifs) else "Motif"
                    for motif in motifs]
        else:
            return ["Motif"] * len(feature_annotations)

    @staticmethod
    def _is_highlight_motif(motif, highlight_motifs):
        for highlight_motif in highlight_motifs:
            if motif == highlight_motif:
                return True

            if len(motif[0]) > len(highlight_motif[0]):
                if MotifPerformancePlotHelper.is_sub_motif(highlight_motif, motif):
                    return True

        return False

    @staticmethod
    def is_sub_motif(short_motif, long_motif):
        assert len(long_motif[0]) > len(short_motif[0])

        long_motif_dict = {long_motif[0][i]: long_motif[1][i] for i in range(len(long_motif[0]))}

        for idx, aa in zip(short_motif[0], short_motif[1]):
            if idx in long_motif_dict.keys():
                if long_motif_dict[idx] != aa:
                    return False
            else:
                return False

        return True


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
    def plot_precision_per_tp(file_path, plotting_data, combined_precision, dataset_type, training_set_name,
                              tp_cutoff, motifs_name="motifs", highlight_motifs_name="highlight"):
        # fig = px.scatter(plotting_data,
        #                y="precision", x="training_TP", hover_data=["feature_names"],
        #                range_y=[0, 1.01], color_discrete_sequence=["#74C4C4"],
        #                # stripmode="overlay",
        #                log_x=True,
        #                labels={
        #                    "precision": f"Precision ({dataset_type})",
        #                    "feature_names": "Motif",
        #                    "training_TP": f"True positive predictions ({training_set_name})"
        #                }, template="plotly_white")


        # make 'base figure' with 1 point
        fig = px.scatter(plotting_data, y=[0], x=[0], range_y=[-0.01, 1.01], log_x=True,
                         template="plotly_white")

        # hide 'base figure' point
        fig.update_traces(marker=dict(size=12, opacity=0), selector=dict(mode='markers'))

        # add data points (needs to be separate trace to show up in legend)
        fig.add_trace(go.Scatter(x=plotting_data["training_TP"], y=plotting_data["precision"],
                                 mode='markers', name="Motif precision",
                                 marker=dict(symbol="circle", color="#74C4C4")),
                      secondary_y=False)

        # add combined precision
        fig.add_trace(go.Scatter(x=combined_precision["training_TP"], y=combined_precision["combined_precision"],
                                 mode='markers+lines', name="Combined precision",
                                 marker=dict(symbol="diamond", color=px.colors.diverging.Tealrose[0])),
                      secondary_y=False)

        # add highlighted motifs
        plotting_data_highlight = plotting_data[plotting_data["highlight"] != "Motif"]
        if len(plotting_data_highlight) > 0:
            fig.add_trace(go.Scatter(x=plotting_data_highlight["training_TP"], y=plotting_data_highlight["precision"],
                                     mode='markers', name=f"{highlight_motifs_name} precision",
                                     marker=dict(symbol="circle", color="#F5C144")),
                          secondary_y=False)

        # add smoothed combined precision
        if "smooth_combined_precision" in combined_precision:
            fig.add_trace(go.Scatter(x=combined_precision["training_TP"], y=combined_precision["smooth_combined_precision"],
                                     marker=dict(color=px.colors.diverging.Tealrose[-1]),
                                     name="Combined precision, smoothed",
                                     mode="lines", line_shape='spline', line={'smoothing': 1.3}),
                          secondary_y=False, )

        # add vertical TP cutoff line
        if tp_cutoff is not None:
            if tp_cutoff == "auto":
                tp_cutoff = min(plotting_data["training_TP"])

            fig.add_vline(x=tp_cutoff, line_dash="dash")

        tickvals = MotifPerformancePlotHelper._get_log_x_axis_ticks(plotting_data, tp_cutoff)
        fig.update_layout(xaxis=dict(tickvals=tickvals),
                          xaxis_title=f"True positive predictions ({training_set_name})",
                          yaxis_title=f"Precision ({dataset_type})",
                          showlegend=True)

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision scores on the {dataset_type} for {motifs_name} found at each true positive count of the {training_set_name}.",
        )

    @staticmethod
    def _get_log_x_axis_ticks(plotting_data, tp_cutoff):
        ticks = []

        min_val, max_val = min(plotting_data["training_TP"]), max(plotting_data["training_TP"])

        i = 1
        while i < max_val:
            if i > min_val:
                ticks.append(i)
            i *= 10

        ticks.append(min_val)
        ticks.append(max_val)

        if tp_cutoff is not None:
            ticks.append(tp_cutoff)

        return sorted(ticks)

    @staticmethod
    def plot_precision_recall(file_path, plotting_data, min_recall=None, min_precision=None, dataset_type=None, motifs_name="motifs",
                              highlight_motifs_name="highlight"):
        fig = px.scatter(plotting_data,
                         y="precision", x="recall", hover_data=["feature_names"],
                         range_x=[0, 1.01], range_y=[0, 1.01], color="highlight",
                         color_discrete_map={"Motif": px.colors.qualitative.Pastel[0],
                                             highlight_motifs_name: px.colors.qualitative.Pastel[1]},
                         labels={
                             "precision": f"Precision ({dataset_type})",
                             "recall": f"Recall ({dataset_type})",
                             "feature_names": "Motif",
                         }, template="plotly_white")

        if min_precision is not None and min_precision > 0:
            fig.add_hline(y=min_precision, line_dash="dash")

        if min_recall is not None and min_recall > 0:
            fig.add_vline(x=min_recall, line_dash="dash")

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision versus recall of significant {motifs_name} on the {dataset_type}",
        )

    @staticmethod
    def write_output_tables(report_obj, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, motifs_name="motifs", file_suffix=""):
        results_table_name = f"Confusion matrix and precision/recall scores for significant {motifs_name}" + " on the {} set"
        combined_precision_table_name = f"Combined precision scores of {motifs_name}" + " on the {} set for each TP value on the " + str(report_obj.training_set_name)

        train_results_table = report_obj._write_output_table(training_plotting_data, report_obj.result_path / f"training_set_scores{file_suffix}.csv", results_table_name.format(report_obj.training_set_name))
        test_results_table = report_obj._write_output_table(test_plotting_data, report_obj.result_path / f"test_set_scores{file_suffix}.csv", results_table_name.format(report_obj.test_set_name))
        training_combined_precision_table = report_obj._write_output_table(training_combined_precision, report_obj.result_path / f"training_combined_precision{file_suffix}.csv", combined_precision_table_name.format(report_obj.training_set_name))
        test_combined_precision_table = report_obj._write_output_table(test_combined_precision, report_obj.result_path / f"test_combined_precision{file_suffix}.csv", combined_precision_table_name.format(report_obj.test_set_name))

        return [table for table in [train_results_table, test_results_table, training_combined_precision_table, test_combined_precision_table] if table is not None]

    @staticmethod
    def write_plots(report_obj, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, training_tp_cutoff, test_tp_cutoff, motifs_name="motifs", file_suffix=""):
        training_tp_plot = report_obj._safe_plot(plot_callable="_plot_precision_per_tp", plotting_data=training_plotting_data, combined_precision=training_combined_precision, dataset_type=report_obj.training_set_name, file_path=report_obj.result_path / f"training_precision_per_tp{file_suffix}.html", motifs_name=motifs_name, tp_cutoff=training_tp_cutoff)
        test_tp_plot = report_obj._safe_plot(plot_callable="_plot_precision_per_tp", plotting_data=test_plotting_data, combined_precision=test_combined_precision, dataset_type=report_obj.test_set_name, file_path=report_obj.result_path / f"test_precision_per_tp{file_suffix}.html", motifs_name=motifs_name, tp_cutoff=test_tp_cutoff)
        training_pr_plot = report_obj._safe_plot(plot_callable="_plot_precision_recall", plotting_data=training_plotting_data, dataset_type=report_obj.training_set_name, file_path=report_obj.result_path / f"training_precision_recall{file_suffix}.html", motifs_name=motifs_name)
        test_pr_plot = report_obj._safe_plot(plot_callable="_plot_precision_recall", plotting_data=test_plotting_data, dataset_type=report_obj.test_set_name, file_path=report_obj.result_path / f"test_precision_recall{file_suffix}.html", motifs_name=motifs_name)

        return [plot for plot in [training_tp_plot, test_tp_plot, training_pr_plot, test_pr_plot] if plot is not None]
