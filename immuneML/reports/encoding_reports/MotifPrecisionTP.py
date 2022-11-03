from pathlib import Path

from typing import List
import logging
import plotly.express as px
import plotly.graph_objects as go

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.encodings.motif_encoding.PositionalMotifHelper import PositionalMotifHelper
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class MotifPrecisionTP(EncodingReport):
    """
    # todo maybe: custom highlight motif name
    # todo maybe: x axis log scale

    # todo refactor to share code with MotifGeneralizationAnalysis

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = MotifPrecisionTP.__name__

        if "highlight_motifs_path" in kwargs and kwargs["highlight_motifs_path"] is not None:
            PositionalMotifHelper.check_motif_filepath(kwargs["highlight_motifs_path"], location, "highlight_motifs_path")

        return MotifPrecisionTP(**kwargs)

    def __init__(self, highlight_motifs_path: str = None, dataset: SequenceDataset = None,
                 result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)
        self.highlight_motifs_path = Path(highlight_motifs_path) if highlight_motifs_path is not None else None
        self.highlight_motifs = None

        if self.highlight_motifs_path is not None:
            self.highlight_motifs = [PositionalMotifHelper.motif_to_string(indices, amino_acids, motif_sep="-", newline=False)
                                     for indices, amino_acids in PositionalMotifHelper.read_motifs_from_file(highlight_motifs_path)]

    def _generate(self):
        PathBuilder.build(self.result_path)

        results_table = self._write_output()
        results_plots = self._safe_plot()

        return ReportResult(output_tables=[results_table], output_figures=results_plots)

    def _write_output(self):
        file_path = self.result_path / f"motif_precision_recall_tp.csv"
        self.dataset.encoded_data.feature_annotations.to_csv(file_path)

        return ReportOutput(
            path=file_path,
            name=f"Precision, recall and number of true positives for each significant motif",
        )

    def _get_color(self, feature_annotations):
        highlight_label = self.highlight_motifs_path.stem.replace("_", " ").capitalize()

        if self.highlight_motifs is not None:
            return [highlight_label if motif in self.highlight_motifs else "Significant motif" for motif in feature_annotations["feature_names"]]
            # return [1 if motif in self.highlight_motifs else 0 for motif in feature_annotations["feature_names"]]

    def _plot_precision_recall(self):
        file_path = self.result_path / f"precision_recall.html"
        feature_annotations = self.dataset.encoded_data.feature_annotations

        y = "precision_scores" if "precision_scores" in feature_annotations.columns else "weighted_precision_scores"
        x = "recall_scores" if "recall_scores" in feature_annotations.columns else "weighted_recall_scores"

        fig = px.scatter(feature_annotations,
                         y=y, x=x, hover_data=["feature_names"],
                         range_x=[0, 1], range_y=[0, 1], color=self._get_color(feature_annotations),
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         labels={
                             "precision_scores": "Precision",
                             "recall_scores": "Recall",
                             "weighted_precision_scores": "Weighted precision",
                             "weighted_recall_scores": "Weighted recall",
                             "feature_names": "Motif"
                         })

        info = self.dataset.encoded_data.info

        if "min_recall" in info and info["min_recall"] is not None:
            fig.add_hline(y=info["min_precision"], line_dash="dash")

        if "min_precision" in info and info["min_precision"] is not None:
            fig.add_vline(x=info["min_recall"], line_dash="dash")

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision versus recall of significant motifs",
        )

    def _plot_precision_per_tp(self):
        file_path = self.result_path / f"precision_per_tp.html"
        feature_annotations = self.dataset.encoded_data.feature_annotations

        y = "precision_scores" if "precision_scores" in feature_annotations.columns else "weighted_precision_scores"

        fig = px.strip(feature_annotations,
                         y=y, x="raw_tp_count", hover_data=["feature_names"],
                         range_y=[0, 1],  color=self._get_color(feature_annotations),
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         stripmode='overlay',
                         labels={
                             "precision_scores": "Precision",
                             "weighted_precision_scores": "Weighted precision",
                             "feature_names": "Motif",
                             "raw_tp_count": "True positive predictions"
                         })


        mean_precision = feature_annotations.groupby("raw_tp_count")[y].mean()

        fig.add_trace(go.Scatter(x=list(mean_precision.index), y=list(mean_precision),
                                 marker=dict(color=px.colors.diverging.Tealrose[0])), secondary_y=False)
        fig.update_layout(showlegend=False)

        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1
            )
        )

        fig.write_html(str(file_path))

        return ReportOutput(
            path=file_path,
            name=f"Precision scores for motifs found at each true positive rate.",
        )

    def _plot(self) -> List[ReportOutput]:
        pr_plot = self._plot_precision_recall()
        tp_plot = self._plot_precision_per_tp()

        return [plot for plot in [pr_plot, tp_plot] if plot is not None]

    def check_prerequisites(self) -> bool:
        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning(f"{MotifPrecisionTP.__name__}: the dataset is not encoded, skipping this report...")
            return False
        elif self.dataset.encoded_data.encoding != MotifEncoder.__name__:
            logging.warning(
                f"{MotifPrecisionTP.__name__}: the dataset encoding ({self.dataset.encoded_data.encoding}) "
                f"does not match the required encoding ({MotifEncoder.__name__}), skipping this report...")
            return False
        elif self.dataset.encoded_data.feature_annotations is None:
            logging.warning(f"{MotifPrecisionTP.__name__}: missing feature annotations for {MotifEncoder.__name__},"
                            f"skipping this report...")
            return False
        else:
            return True