import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class NodeDegreeDistribution(DataReport):

    def build_object(cls, **kwargs):
        #TODO: Add validation of kwargs
        return NodeDegreeDistribution(**kwargs)

    def __init__(self, dataset: SequenceDataset = None, result_path: Path = None, name: str = None,
                 compairr_path: str = None, indels: bool = False, ignore_counts: bool = False,
                 ignore_genes: bool = False, hamming_distance: int = 1, threads: int = 4):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.compairr_path = compairr_path
        self.indels = indels
        self.ignore_counts = ignore_counts
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

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        #TODO: Deduplicate

        node_degree_distribution = self._compute_node_degree_distribution()

        return ReportResult()

    def _compute_node_degree_distribution(self):
        compairr_existence_output = self._run_compairr()

        if compairr_existence_output is None:
            raise RuntimeError("CompAIRR execution failed: no output returned. ")

        if compairr_existence_output.empty:
            return pd.Series(dtype=int)

        compairr_existence_output["dataset_1"] -= 1
        node_degree_distribution = compairr_existence_output["dataset_1"].value_counts()

        return node_degree_distribution

    def _run_compairr(self):
        output_file = self.result_path / f"compairr_existence.tsv"

        cmd_args = [str(self.compairr_path), "--existence", str(self.dataset.dataset_file),
                    str(self.dataset.dataset_file), "-o", str(output_file), "-d", str(self.hamming_distance)]
        if self.indels:
            cmd_args.append("-i")
        if self.ignore_counts:
            cmd_args.append("--ignore-counts")
        if self.ignore_genes:
            cmd_args.append("--ignore-genes")
        cmd_args.extend(["-t", str(self.threads)])

        subprocess.run(cmd_args, capture_output=True, text=True)

        compairr_existence_output = None
        if output_file.is_file():
            if os.path.getsize(output_file) > 0:
                compairr_existence_output = pd.read_csv(output_file, sep="\t")
            else:
                compairr_existence_output = pd.DataFrame()
            os.remove(str(output_file))

        return compairr_existence_output
