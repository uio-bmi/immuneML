from multiprocessing.pool import Pool
from pathlib import Path
import os
import numpy as np
import pandas as pd
import logging

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
import subprocess
import shutil


print("changed!!")

class TCRMatchEpitopeAnalysis(DataReport):
    """

    Arguments:

        compairr_path (str):

        tcrmatch_path (str):

        iedb_file (str):

        differences (int):

        indels (bool):

        threads (int):





    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        reports:
            my_analysis: TCRMatchEpitopeAnalysis

    """

    def __init__(self, dataset: Dataset = None, result_path: Path = None, iedb_file: Path = None,
                 compairr_path: Path = None, tcrmatch_path: Path = None,
                 differences: int = None, indels: bool = None, threads: int = None, chunk_size: int = 100000,
                 threshold: float = None, keep_tmp_results: bool = None,
                 number_of_processes: int = 1, name: str = None):
        print("inside init")
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.compairr_path = compairr_path
        self.tcrmatch_path = tcrmatch_path
        self.iedb_file = iedb_file
        self.differences = differences
        self.indels = indels
        self.threshold = threshold
        self.threads = threads
        self.chunk_size = chunk_size
        self.keep_tmp_results = keep_tmp_results
        #

    @classmethod
    def build_object(cls, **kwargs):
        location = TCRMatchEpitopeAnalysis.__name__
        print("inside build object")

        assert "iedb_file" in kwargs, f"{location}: expected iedb_file to be set for {location} report"
        ParameterValidator.assert_type_and_value(kwargs["iedb_file"], str, location, "iedb_file")
        ParameterValidator.assert_valid_tabular_file(kwargs["iedb_file"], location, "iedb_file", sep="\t",
                                                     expected_columns=["cdr3_aa", "original_seq", "receptor_group",
                                                                       "epitopes", "source_organisms", "source_antigens"])
        kwargs["iedb_file"] = Path(kwargs["iedb_file"])

        kwargs["compairr_path"] = Path(CompAIRRHelper.determine_compairr_path(kwargs["compairr_path"], required_major=1, required_minor=11, required_patch=0))

        TCRMatchEpitopeAnalysis.check_tcrmatch_path(kwargs["tcrmatch_path"])
        kwargs["tcrmatch_path"] = Path(kwargs["tcrmatch_path"])

        ParameterValidator.assert_type_and_value(kwargs["differences"], int, location, "differences", min_inclusive=0)
        ParameterValidator.assert_type_and_value(kwargs["indels"], bool, location, "indels")
        ParameterValidator.assert_type_and_value(kwargs["keep_tmp_results"], bool, location, "keep_tmp_results")
        ParameterValidator.assert_type_and_value(kwargs["threads"], int, location, "threads", min_inclusive=1)
        ParameterValidator.assert_type_and_value(kwargs["threshold"], float, location, "threshold", min_exclusive=0, max_exclusive=1)

        return TCRMatchEpitopeAnalysis(**kwargs)

    @staticmethod
    def check_tcrmatch_path(tcrmatch_path):
        try:
            p = subprocess.run([str(tcrmatch_path)], shell=True, capture_output=True)
            returncode = p.returncode
        except Exception:
            returncode = None

        assert returncode != 127, f"{TCRMatchEpitopeAnalysis.__name__}: tcrmatch_path not found (exit code 127): {tcrmatch_path}"

    def _generate(self) -> ReportResult:
        print("printing tcrmatch path")

        self.tcrmatch_files_path = PathBuilder.build(self.result_path / "tcrmatch_results_per_repertoire")
        print(self.tcrmatch_files_path)
        logging.info(f"built tcrmatch file path at {self.tcrmatch_files_path}")


        epitope_matches = self._run_tcrmatch_pipeline()

        return ReportResult(name=self.name,
                            info="test",
                            output_text=[ReportOutput(self.tcrmatch_files_path, f"dafdaf")])

    def _run_tcrmatch_pipeline(self):
        with Pool(processes=self.number_of_processes) as pool:
            epitope_matches = pool.map(self._run_tcrmatch_pipeline_for_repertoire, self.dataset.get_data())

        return epitope_matches

    def _run_tcrmatch_pipeline_for_repertoire(self, repertoire: Repertoire):
        # export something from the reprtoire: cdr3s without leading/trailing



        repertoire_result_path = PathBuilder.build(self.result_path / f"results_per_repertoire/{repertoire.identifier}")
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: made rep results path {repertoire_result_path}")

        tcrmatch_input_files_path = PathBuilder.build(repertoire_result_path / "tcrmatch_input_files")
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: made tcrmatch results path {tcrmatch_input_files_path}")

        repertoire_output_file_path = self.tcrmatch_files_path / f"rep_{repertoire.identifier}.tsv"

        cdr3s_file = repertoire_result_path / "cdr3_aas.txt"
        self._export_repertoire_cdr3s(cdr3s_file, repertoire)
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: rep cdr3s exported to {cdr3s_file}")

        pairs_file = self._create_pairs_file_with_compairr(repertoire_result_path, cdr3s_file)
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: pairs file at {pairs_file}")

        self._make_tcrmatch_input_files(pairs_file, tcrmatch_input_files_path)

        self._run_tcrmatch_on_each_file(tcrmatch_input_files_path, repertoire_output_file_path)

        if not self.keep_tmp_results:
            shutil.rmtree(repertoire_result_path)

        return repertoire_output_file_path

    def _export_repertoire_cdr3s(self, filename, repertoire: Repertoire):
        np.savetxt(fname=filename, X=repertoire.get_sequence_aas(), header="cdr3_aa", comments="", fmt="%s")

    def _create_pairs_file_with_compairr(self, compairr_result_path, repertoire_cdr3s_file):
        pairs_file = compairr_result_path / "pairs.txt"
        compairr_log = compairr_result_path / "log.txt"

        cmd_args = [str(self.compairr_path), str(self.iedb_file), str(repertoire_cdr3s_file), "--matrix",
                    "--differences", str(self.differences), "--ignore-counts", "--ignore-genes",
                    "--cdr3", "--pairs", str(pairs_file), "--threads", str(self.threads),
                    "--log", str(compairr_log),
                    "--output", str(compairr_result_path / "out.txt"),
                    "--keep-columns", "original_seq,receptor_group,epitopes,source_organisms,source_antigens"]

        indels_args = ["--indels"] if self.indels else []
        cmd_args += indels_args

        subprocess_result = subprocess.run(cmd_args, capture_output=True, text=True, check=True)

        if not pairs_file.is_file():
            err_str = f": {subprocess_result.stderr}" if subprocess_result.stderr else ""

            raise RuntimeError(f"An error occurred while running CompAIRR{err_str}\n"
                               f"See the log file for more information: {compairr_log}")

        if os.path.getsize(pairs_file) == 0:
            raise RuntimeError("An error occurred while running CompAIRR: output pairs file is empty.\n"
                               f"See the log file for more information: {compairr_log}")

        return pairs_file

    def _export_cdr3(self, export_cdr3, output_file):
        with open(output_file, "w") as file:
            file.write(f"{export_cdr3}\n")

    def _make_tcrmatch_input_files(self, pairs_file, output_folder):
        IEDB_COLUMNS = ["trimmed_seq", "original_seq", "receptor_group", "epitopes", "source_organisms", "source_antigens"]
        COLUMN_ORDER = ["cdr3_aa_1", "original_seq_1", "receptor_group_1", "epitopes_1", "source_organisms_1", "source_antigens_1"]

        df = pd.read_csv(pairs_file, sep="\t",
                         usecols=["cdr3_aa_1", "cdr3_aa_2", "original_seq_1", "receptor_group_1", "epitopes_1",
                                  "source_organisms_1", "source_antigens_1"], iterator=True, chunksize=self.chunk_size)

        existing_files = {}
        counter = 0

        for chunk in df:
            for user_cdr3, cdr3_chunk in chunk.groupby("cdr3_aa_2"):
                if user_cdr3 in existing_files:
                    id = existing_files[user_cdr3]
                    cdr3_chunk[COLUMN_ORDER].to_csv(f"{output_folder}/prefiltered_IEDB_{id}.tsv", sep="\t", mode="a",
                                                    index=False, header=False)

                else:
                    counter += 1
                    existing_files[user_cdr3] = counter
                    id = counter

                    cdr3_chunk[COLUMN_ORDER].to_csv(f"{output_folder}/prefiltered_IEDB_{id}.tsv", sep="\t", index=False,
                                                    header=IEDB_COLUMNS)
                    self._export_cdr3(user_cdr3, f"{output_folder}/user_cdr3_{id}.tsv")

    def _run_tcrmatch_on_each_file(self, tcrmatch_input_path, output_file_path):
        TCRMATCH_HEADER = "input_sequence\tmatch_sequence\tscore\treceptor_group\tepitope\tantigen\torganism\t"
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: inside run tcrmatch on each file ")

        with open(output_file_path, "w") as output_file:
            output_file.write(TCRMATCH_HEADER + "\n")

            for iedb_file in tcrmatch_input_path.glob("prefiltered_IEDB_*.tsv"):
                logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: running with iedb file {iedb_file}")

                id = iedb_file.stem.split("_")[-1]
                user_file = tcrmatch_input_path / f"user_cdr3_{id}.tsv"

                assert user_file.is_file(), f"Found iedb file {iedb_file} but not the matching user cdr3 file {user_file}."

                cmd_args = [str(self.tcrmatch_path), "-i", str(user_file), "-t", "1", "-d", str(iedb_file), "-s", str(self.threshold)]

                logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: running cmd args: {cmd_args}")

                subprocess_result = subprocess.run(cmd_args, capture_output=True, text=True, check=True)

                if subprocess_result.stdout == "":
                    err_str = f":{subprocess_result.stderr}"
                    raise RuntimeError(f"An error occurred while running TCRMatch{err_str}\n"
                                       f"The following arguments were used: {' '.join(cmd_args)}")

                header, content = subprocess_result.stdout.split("\n", 1)

                assert header == TCRMATCH_HEADER, f"TCRMatch result does not contain the expected header.\n" \
                                                  f"Expected header: {TCRMATCH_HEADER}\n" \
                                                  f"Found instead: {header}\n" \
                                                  f"The following arguments were used: {' '.join(cmd_args)}"

                output_file.write(content)

    def check_prerequisites(self):
        # todo check if repertoire dataset
        pass
