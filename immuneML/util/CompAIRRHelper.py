import os
import subprocess
import warnings
from pathlib import Path

import pandas as pd

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.CompAIRRParams import CompAIRRParams


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
    def get_cmd_args(compairr_params: CompAIRRParams, input_file_list, result_path):
        indels_args = ["-i"] if compairr_params.indels else []
        frequency_args = ["-f"] if compairr_params.ignore_counts else []
        ignore_genes = ["-g"] if compairr_params.ignore_genes else []
        output_args = ["-o", str(result_path / compairr_params.output_filename), "-l", str(result_path / compairr_params.log_filename)]
        output_pairs = ['-p', str(result_path / compairr_params.pairs_filename)] if compairr_params.output_pairs else []
        cdr3_indicator = ['--cdr3'] if compairr_params.is_cdr3 else []
        command = '-m' if compairr_params.do_repertoire_overlap and not compairr_params.do_sequence_matching else '-x'

        return [str(compairr_params.compairr_path), command, "-d", str(compairr_params.differences), "-t", str(compairr_params.threads)] + \
               indels_args + frequency_args + ignore_genes + output_args + [str(file) for file in input_file_list] + output_pairs + cdr3_indicator

    @staticmethod
    def write_repertoire_file(repertoire_dataset=None, filename=None, compairr_params=None, repertoires: list = None,
                              export_sequence_id: bool = False):
        mode = "w"
        header = True

        columns_in_order = []

        if repertoire_dataset is not None and repertoires is None:
            repertoires = repertoire_dataset.get_data()

        for ind, repertoire in enumerate(repertoires):
            repertoire_contents = CompAIRRHelper.get_repertoire_contents(repertoire, compairr_params, export_sequence_id)

            if ind == 0:
                columns_in_order = sorted(repertoire_contents.columns)

            repertoire_contents[columns_in_order].to_csv(filename, mode=mode, header=header, index=False, sep="\t")

            mode = "a"
            header = False

    @staticmethod
    def write_sequences_file(sequence_dataset, filename, compairr_params, repertoire_id="sequence_dataset"):
        compairr_data = {"junction_aa": [],
                         "repertoire_id": [],
                         "sequence_id": []}

        if not compairr_params.ignore_genes:
            compairr_data["v_call"] = []
            compairr_data["j_call"] = []

        if not compairr_params.ignore_counts:
            compairr_data["duplicate_count"] = []

        for sequence in sequence_dataset.get_data():
            compairr_data["junction_aa"].append(sequence.get_sequence(sequence_type=SequenceType.AMINO_ACID))

            assert sequence.identifier is not None, f"{CompAIRRHelper.__name__}: sequence identifiers must be set when exporting a sequence dataset for CompAIRR"
            compairr_data["sequence_id"].append(sequence.identifier)
            compairr_data["repertoire_id"].append(repertoire_id)

            if not compairr_params.ignore_genes:
                compairr_data["v_call"].append(sequence.get_attribute("v_gene"))
                compairr_data["j_call"].append(sequence.get_attribute("j_gene"))

            if not compairr_params.ignore_counts:
                compairr_data["duplicate_count"].append(sequence.get_attribute("count"))

        df = pd.DataFrame(compairr_data)

        df.to_csv(filename, mode="w", header=True, index=False, sep="\t")

    @staticmethod
    def get_repertoire_contents(repertoire, compairr_params, export_sequence_id=False):
        attributes = [EnvironmentSettings.get_sequence_type().value, "duplicate_count"]
        attributes += [] if compairr_params.ignore_genes else ["v_call", "j_call"]
        repertoire_contents = repertoire.get_attributes(attributes)
        repertoire_contents = pd.DataFrame({**repertoire_contents, "identifier": repertoire.identifier})
        if export_sequence_id:
            repertoire_contents['sequence_id'] = repertoire.get_attribute('sequence_id')

        check_na_rows = [EnvironmentSettings.get_sequence_type().value]
        check_na_rows += [] if compairr_params.ignore_counts else ["duplicate_count"]
        check_na_rows += [] if compairr_params.ignore_genes else ["v_call", "j_call"]

        n_rows_before = len(repertoire_contents)

        repertoire_contents.dropna(inplace=True, subset=check_na_rows)

        if n_rows_before > len(repertoire_contents):
            warnings.warn(
                f"CompAIRRHelper: removed {n_rows_before - len(repertoire_contents)} entries from repertoire {repertoire.identifier} due to missing values.")

        if compairr_params.ignore_counts:
            repertoire_contents["duplicate_count"] = 1
        else:
            repertoire_contents['duplicate_count'] = [count if count >= 0 else pd.NA for count in repertoire_contents['duplicate_count']]

        repertoire_contents.rename(columns={EnvironmentSettings.get_sequence_type().value: "cdr3_aa" if repertoire.get_region_type() == RegionType.IMGT_CDR3 else 'junction_aa',
                                            "identifier": "repertoire_id"},
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
