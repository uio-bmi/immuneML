import os
import subprocess
import warnings
from pathlib import Path

import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings


class CompAIRRHelper:

    @staticmethod
    def determine_compairr_path(compairr_path):
        if compairr_path is None:
            try:
                compairr_path = CompAIRRHelper.check_compairr_path("compairr")
            except Exception as e:
                compairr_path = CompAIRRHelper.check_compairr_path("/usr/local/bin/compairr")
        else:
            compairr_path = CompAIRRHelper.check_compairr_path(compairr_path)

        return compairr_path

    @staticmethod
    def check_compairr_path(compairr_path):
        required_major = 1
        required_minor = 3
        required_patch = 2

        try:
            compairr_result = subprocess.run([str(Path(compairr_path)), "--version"], capture_output=True)
            assert compairr_result.returncode == 0, "exit code was non-zero."
            output = str(compairr_result.stderr).split()
            major, minor, patch = output[1].split(".")

            mssg = f"CompAIRR version {required_major}.{required_minor}.{required_patch} or higher is required, found version {output[1]}"
            assert int(major) >= required_major, mssg
            if major == 1:
                assert int(minor) >= required_minor, mssg
                if minor == 3:
                    assert int(patch) >= {required_patch}, mssg
        except Exception as e:
            raise Exception(f"CompAIRRHelper: failed to call CompAIRR: {e}\n"
                            f"Please ensure the correct version of CompAIRR has been installed (version {required_major}.{required_minor}.{required_patch} or later), "
                            f"or provide the path to the CompAIRR executable.")

        return compairr_path

    @staticmethod
    def get_cmd_args(compairr_params, input_file_list, result_path):
        indels_args = ["-i"] if compairr_params.indels else []
        frequency_args = ["-f"] if compairr_params.ignore_counts else []
        ignore_genes = ["-g"] if compairr_params.ignore_genes else []
        output_args = ["-o", str(result_path / compairr_params.output_filename), "-l", str(result_path / compairr_params.log_filename)]

        return [str(compairr_params.compairr_path), "-m", "-d", str(compairr_params.differences), "-t", str(compairr_params.threads)] + \
               indels_args + frequency_args + ignore_genes + output_args + input_file_list

    @staticmethod
    def write_repertoire_file(repertoire_dataset, filename, compairr_params):
        mode = "w"
        header = True

        for repertoire in repertoire_dataset.get_data():
            repertoire_contents = CompAIRRHelper.get_repertoire_contents(repertoire, compairr_params)
            repertoire_contents.to_csv(filename, mode=mode, header=header, index=False, sep="\t")

            mode = "a"
            header = False

    @staticmethod
    def get_repertoire_contents(repertoire, compairr_params):
        attributes = [EnvironmentSettings.get_sequence_type().value, "counts"]
        attributes += [] if compairr_params.ignore_genes else ["v_genes", "j_genes"]
        repertoire_contents = repertoire.get_attributes(attributes)
        repertoire_contents = pd.DataFrame({**repertoire_contents, "identifier": repertoire.identifier})

        check_na_rows = [EnvironmentSettings.get_sequence_type().value]
        check_na_rows += [] if compairr_params.ignore_counts else ["counts"]
        check_na_rows += [] if compairr_params.ignore_genes else ["v_genes", "j_genes"]

        n_rows_before = len(repertoire_contents)

        repertoire_contents.dropna(inplace=True, subset=check_na_rows)

        if n_rows_before > len(repertoire_contents):
            warnings.warn(
                f"CompAIRRHelper: removed {n_rows_before - len(repertoire_contents)} entries from repertoire {repertoire.identifier} due to missing values.")

        if compairr_params.ignore_counts:
            repertoire_contents["counts"] = 1

        repertoire_contents.rename(columns={EnvironmentSettings.get_sequence_type().value: "junction_aa",
                                            "v_genes": "v_call", "j_genes": "j_call",
                                            "counts": "duplicate_count", "identifier": "repertoire_id"},
                                   inplace=True)

        return repertoire_contents

    @staticmethod
    def verify_compairr_output_path(subprocess_result, compairr_params, result_path):
        output_file = result_path / compairr_params.output_filename

        if not output_file.is_file():
            raise RuntimeError(
                f"CompAIRRHelper: failed to calculate the distance matrix with CompAIRR ({compairr_params.compairr_path}). "
                f"The following error occurred: {subprocess_result.stderr}")

        if os.path.getsize(output_file) == 0:
            raise RuntimeError(
                f"CompAIRRHelper: failed to calculate the distance matrix with CompAIRR ({compairr_params.compairr_path}), output matrix is empty. "
                f"For details see the log file at {result_path / compairr_params.log_filename}")

        return output_file

    @staticmethod
    def read_compairr_output_file(output_file):
        return pd.read_csv(output_file, sep="\t", index_col=0)

    @staticmethod
    def process_compairr_output_file(subprocess_result, compairr_params, result_path):
        output_file = CompAIRRHelper.verify_compairr_output_path(subprocess_result, compairr_params, result_path)
        return CompAIRRHelper.read_compairr_output_file(output_file)
