import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class NodeDegreeDistribution(DataReport):
    """
    A report that uses CompAIRR to compute the node degree distribution of a sequence dataset. Results are visualized
    as a histogram and stored in a TSV file.

    The report assumes that CompAIRR (https://github.com/uio-bmi/compairr) has been installed.

    **Specification arguments**:

    - dataset (SequenceDataset): The dataset to analyze.

    - compairr_path (str): The path to the CompAIRR executable.

    - region_type (str): The region type to analyze. Can be either "IMGT_CDR3" or "IMGT_JUNCTION".

    - indels (bool): Whether to include indels in the analysis. Default is False.

    - ignore_genes (bool): Whether to ignore gene names in the analysis. Default is False.

    - hamming_distance (int): The Hamming distance to use for the analysis. Default is 1.

    - threads (int): The number of threads to use for the analysis. Default is 4.

    **Yaml specification**:
    .. indent with spaces
    .. code-block:: yaml
        NodeDegreeDistribution:
            dataset: <dataset_name>
            result_path: <path_to_result>
            compairr_path: <path_to_compairr>
            region_type: IMGT_JUNCTION
            indels: False
            ignore_genes: False
            hamming_distance: 1
            threads: 4

    """

    @classmethod
    def build_object(cls, **kwargs):
        ParameterValidator.assert_region_type(kwargs)

        return NodeDegreeDistribution(**{**kwargs, 'region_type': RegionType[kwargs['region_type'].upper()]})

    def __init__(self, dataset: SequenceDataset = None, result_path: Path = None, name: str = None,
                 compairr_path: str = None, region_type: RegionType = RegionType.IMGT_JUNCTION, indels: bool = False,
                 ignore_genes: bool = False, hamming_distance: int = 1, threads: int = 4):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.compairr_path = compairr_path
        self.region_type = region_type
        self.indels = indels
        self.ignore_genes = ignore_genes
        self.hamming_distance = hamming_distance
        self.threads = threads

    def check_prerequisites(self) -> bool:
        if not self.compairr_path:
            logging.warning("CompAIRR path not provided. CompAIRR must be installed and available in the system PATH.")
            return False
        if not isinstance(self.dataset, SequenceDataset):
            logging.warning(f"{NodeDegreeDistribution.__name__} report can only be generated for SequenceDataset.")
            return False
        if self.region_type not in {RegionType.IMGT_CDR3, RegionType.IMGT_JUNCTION}:
            logging.warning(f"Invalid region type: {self.region_type.value.upper()}. "
                            f"{NodeDegreeDistribution.__name__} report can only be generated for "
                            f"{RegionType.IMGT_CDR3.value.upper()} or {RegionType.IMGT_JUNCTION.value.upper()}.")
            return False
        return True

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        node_degree_dist = self._compute_node_degree_dist()
        node_degree_dist_output = self._store_node_degree_dist(node_degree_dist, "node_degree_distribution")
        node_degree_dist_histogram = self._plot_node_degree_dist(node_degree_dist, "node_degree_distribution")

        return ReportResult(
            name=self.name,
            info=f"Node degree distribution with hamming distance {self.hamming_distance}",
            output_figures=[node_degree_dist_histogram],
            output_tables=[node_degree_dist_output]
        )

    def _compute_node_degree_dist(self):
        compairr_existence_output = self._run_compairr()

        if compairr_existence_output is None:
            raise RuntimeError("CompAIRR execution failed to generate output.")

        compairr_existence_output["overlap_count"] -= 1
        node_degree_dist = compairr_existence_output["overlap_count"].value_counts()

        return node_degree_dist

    def _run_compairr(self):
        output_file = self.result_path / f"compairr_existence.tsv"

        dedupliacted_dataset_filename = self._deduplicate_sequences()
        cmd_args = [str(self.compairr_path), "--existence", str(dedupliacted_dataset_filename),
                    str(dedupliacted_dataset_filename), "--output", str(output_file), "--differences",
                    str(self.hamming_distance), "--ignore-counts"]
        if self.indels:
            cmd_args.append("--indels")
        if self.ignore_genes:
            cmd_args.append("--ignore-genes")
        if self.region_type == RegionType.IMGT_CDR3:
            cmd_args.append("--cdr3")

        cmd_args.extend(["-t", str(self.threads)])

        subprocess.run(cmd_args, capture_output=True, text=True)

        compairr_existence_output = None
        if output_file.is_file() and os.path.getsize(output_file) > 0:
            compairr_existence_output = pd.read_csv(output_file, sep='\t', names=['sequence_id', 'overlap_count'],
                                                    header=0)
            os.remove(str(output_file))

        os.remove(dedupliacted_dataset_filename)

        return compairr_existence_output

    def _deduplicate_sequences(self):
        deduplicated_dataset_filename = self.dataset.filename.with_name(
            f"{self.dataset.filename.stem}_deduplicated{self.dataset.filename.suffix}"
        )
        dataset = pd.read_csv(self.dataset.filename, sep="\t", header=0)
        subset = self._get_deduplication_subset()

        deduplicated_dataset = dataset.drop_duplicates(subset=subset)
        deduplicated_dataset.to_csv(deduplicated_dataset_filename, sep="\t", index=False)

        return deduplicated_dataset_filename

    def _get_deduplication_subset(self):
        if self.region_type == RegionType.IMGT_CDR3:
            subset = ["cdr3_aa"]
        elif self.region_type == RegionType.IMGT_JUNCTION:
            subset = ["junction_aa"]
        else:
            raise ValueError(f"Unsupported region type: {self.region_type}")

        if not self.ignore_genes:
            subset.extend(["v_call", "j_call"])

        return subset

    def _store_node_degree_dist(self, node_degree_dist: pd.DataFrame, name: str) -> ReportOutput:
        output_path = self.result_path / f"{name}.tsv"
        node_degree_dist.to_csv(output_path, sep="\t")
        return ReportOutput(path=output_path, name=name)

    def _plot_node_degree_dist(self, node_degree_dist: pd.DataFrame, name: str) -> ReportOutput:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=node_degree_dist.index, y=node_degree_dist.values))
        fig.update_layout(xaxis_title="Degree",
                          yaxis_title="Count")
        output_path = self.result_path / f"{name}.html"
        fig.write_html(output_path)

        return ReportOutput(path=output_path, name=name)
