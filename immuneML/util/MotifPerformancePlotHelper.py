
import warnings
from scipy.stats import lognorm
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.MotifPerformanceParams import MotifPerformanceParams


class MotifPerformancePlotHelper():

    @staticmethod
    def get_plotting_data(training_encoded_data, test_encoded_data, params: MotifPerformanceParams):
        training_feature_annotations = MotifPerformancePlotHelper._get_annotated_feature_annotations(training_encoded_data, params)
        test_feature_annotations = MotifPerformancePlotHelper._get_annotated_feature_annotations(test_encoded_data, params)

        training_feature_annotations["training_TP"] = training_feature_annotations["TP"]
        test_feature_annotations = MotifPerformancePlotHelper.merge_train_test_feature_annotations(training_feature_annotations, test_feature_annotations)

        return training_feature_annotations, test_feature_annotations

    @staticmethod
    def _get_annotated_feature_annotations(encoded_data, params: MotifPerformanceParams):
        feature_annotations = encoded_data.feature_annotations.copy()
        MotifPerformancePlotHelper._annotate_confusion_matrix(feature_annotations)
        MotifPerformancePlotHelper._annotate_highlight(feature_annotations, params.highlight_motifs_path, params.highlight_motifs_name)

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
    def get_report_outputs(training_plotting_data, test_plotting_data, params: MotifPerformanceParams):
        if params.split_by_motif_size:
            return MotifPerformancePlotHelper._construct_and_plot_data_per_motif_size(training_plotting_data, test_plotting_data, params=params)
        else:
            return MotifPerformancePlotHelper._construct_and_plot_data(training_plotting_data, test_plotting_data, params=params)

    @staticmethod
    def _construct_and_plot_data_per_motif_size(training_plotting_data, test_plotting_data, params):
        output_plots, output_tables, output_texts = [], [], []

        training_plotting_data["motif_size"] = training_plotting_data["feature_names"].apply(PositionalMotifHelper.get_motif_size)
        test_plotting_data["motif_size"] = test_plotting_data["feature_names"].apply(PositionalMotifHelper.get_motif_size)

        for motif_size in sorted(set(training_plotting_data["motif_size"])):
            sub_training_plotting_data = training_plotting_data[training_plotting_data["motif_size"] == motif_size]
            sub_test_plotting_data = test_plotting_data[test_plotting_data["motif_size"] == motif_size]

            sub_output_tables, sub_output_texts, sub_output_plots = MotifPerformancePlotHelper._construct_and_plot_data(sub_training_plotting_data, sub_test_plotting_data,
                                                                                                                        params, motif_size=motif_size)
            output_plots.extend(sub_output_plots)
            output_tables.extend(sub_output_tables)
            output_texts.extend(sub_output_texts)

        return output_plots, output_tables, output_texts

    @staticmethod
    def _construct_and_plot_data(training_plotting_data, test_plotting_data, params: MotifPerformanceParams, motif_size=None):
        motif_size_suffix = f"_motif_size={motif_size}" if motif_size is not None else ""
        motifs_name = f"motifs of length {motif_size}" if motif_size is not None else "motifs"

        training_combined_precision = MotifPerformancePlotHelper.get_combined_precision(training_plotting_data, params)
        test_combined_precision = MotifPerformancePlotHelper.get_combined_precision(test_plotting_data, params)

        if params.determine_tp_cutoff:
            tp_cutoff = MotifPerformancePlotHelper._determine_tp_cutoff(test_combined_precision, motif_size)
            recall_cutoff = tp_cutoff / params.n_positives_in_training_data
            output_texts = MotifPerformancePlotHelper._write_stats(tp_cutoff, recall_cutoff,
                                                                   motifs_name=motifs_name, file_suffix=motif_size_suffix)
        else:
            tp_cutoff, recall_cutoff, output_texts = None, None, None

        output_tables = MotifPerformancePlotHelper._write_output_tables(training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision,
                                                                        params=params, motifs_name=motifs_name, file_suffix=motif_size_suffix)

        output_plots = MotifPerformancePlotHelper._write_plots(training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, tp_cutoff, motifs_name=motifs_name, file_suffix=motif_size_suffix)

        return output_plots, output_tables, output_texts

    @staticmethod
    def _determine_tp_cutoff(combined_precision, params: MotifPerformanceParams, motif_size=None):
        col = "smooth_combined_precision" if "smooth_combined_precision" in combined_precision else "combined_precision"

        try:
            # assert all(training_combined_precision["training_TP"] == test_combined_precision["training_TP"])
            #
            # train_test_difference = training_combined_precision[col] - test_combined_precision[col]
            # return min(test_combined_precision[train_test_difference <= self.precision_difference]["training_TP"])

            max_tp_below_threshold = max(combined_precision[combined_precision[col] < params.test_precision_threshold]["training_TP"])
            all_above_threshold = combined_precision[combined_precision["training_TP"] > max_tp_below_threshold]

            return min(all_above_threshold["training_TP"])
        except ValueError:
            motif_size_warning = f" for motif size = {motif_size}" if motif_size is not None else ""
            warnings.warn(f"{params.class_name}: could not automatically determine optimal TP threshold{motif_size_warning} with precison differenc  based on {col}")
            return None

    @staticmethod
    def _write_stats(tp_cutoff, recall_cutoff, params, motifs_name="all motifs", file_suffix=""):
        output_path = params.result_path / f"tp_recall_cutoffs{file_suffix}.txt"

        with open(output_path, "w") as file:
            file.writelines([f"total training+test size: {params.dataset_size}\n",
                             f"total positives in training data: {params.n_positives_in_training_data}\n"
                             f"training TP cutoff: {tp_cutoff}\n",
                             f"training recall cutoff: {recall_cutoff}"])

        return [ReportOutput(path=output_path, name=f"TP and recall cutoffs for {motifs_name}")]

    @staticmethod
    def get_combined_precision(plotting_data, params: MotifPerformanceParams):
        group_by_tp = plotting_data.groupby("training_TP")

        combined_precision = group_by_tp["TP"].sum() / (group_by_tp["TP"].sum() + group_by_tp["FP"].sum())

        df = pd.DataFrame({"training_TP": list(combined_precision.index),
                           "combined_precision": list(combined_precision)})

        df["smooth_combined_precision"] = MotifPerformancePlotHelper._smooth_combined_precision(list(combined_precision.index),
                                                                                                    list(combined_precision),
                                                                                                    list(group_by_tp["TP"].count()),
                                                                                                    params)

        return df

    @staticmethod
    def _smooth_combined_precision(x, y, weights, params: MotifPerformanceParams):
        smoothed_y = []

        for i in range(len(x)):
            scale = MotifPerformancePlotHelper._get_lognorm_scale(x, i, weights, params.min_points_in_window, params.smoothing_constant1, params.smoothing_constant2)

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


    @staticmethod
    def _write_output_tables(result_path, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, params, motifs_name="motifs", file_suffix=""):
        results_table_name = f"Confusion matrix and precision/recall scores for significant {motifs_name}" + " on the {} set"
        combined_precision_table_name = f"Combined precision scores of {motifs_name}" + " on the {} set for each TP value on the " + str(params.training_set_name)

        train_results_table = MotifPerformancePlotHelper._write_output_table(training_plotting_data, result_path / f"training_set_scores{file_suffix}.csv", results_table_name.format(params.training_set_name))
        test_results_table = MotifPerformancePlotHelper._write_output_table(test_plotting_data, result_path / f"test_set_scores{file_suffix}.csv", results_table_name.format(params.test_set_name))
        training_combined_precision_table = MotifPerformancePlotHelper._write_output_table(training_combined_precision, result_path / f"training_combined_precision{file_suffix}.csv", combined_precision_table_name.format(params.training_set_name))
        test_combined_precision_table = MotifPerformancePlotHelper._write_output_table(test_combined_precision, result_path / f"test_combined_precision{file_suffix}.csv", combined_precision_table_name.format(params.test_set_name))

        return [table for table in [train_results_table, test_results_table, training_combined_precision_table, test_combined_precision_table] if table is not None]
    @staticmethod
    def _write_output_tables(result_path, training_plotting_data, test_plotting_data, training_combined_precision, test_combined_precision, training_set_name, test_set_name, motifs_name="motifs", file_suffix=""):
        results_table_name = f"Confusion matrix and precision/recall scores for significant {motifs_name}" + " on the {} set"
        combined_precision_table_name = f"Combined precision scores of {motifs_name}" + " on the {} set for each TP value on the " + str(training_set_name)

        train_results_table = MotifPerformancePlotHelper._write_output_table(training_plotting_data, result_path / f"training_set_scores{file_suffix}.csv", results_table_name.format(training_set_name))
        test_results_table = MotifPerformancePlotHelper._write_output_table(test_plotting_data, result_path / f"test_set_scores{file_suffix}.csv", results_table_name.format(test_set_name))
        training_combined_precision_table = MotifPerformancePlotHelper._write_output_table(training_combined_precision, result_path / f"training_combined_precision{file_suffix}.csv", combined_precision_table_name.format(training_set_name))
        test_combined_precision_table = MotifPerformancePlotHelper._write_output_table(test_combined_precision, result_path / f"test_combined_precision{file_suffix}.csv", combined_precision_table_name.format(test_set_name))

        return [table for table in [train_results_table, test_results_table, training_combined_precision_table, test_combined_precision_table] if table is not None]

    @staticmethod
    def _write_output_table(self, table, file_path, name=None):
        sep = "," if file_path.suffix == "csv" else "\t"
        table.to_csv(file_path, index=False, sep=sep)

        return ReportOutput(path=file_path, name=name)