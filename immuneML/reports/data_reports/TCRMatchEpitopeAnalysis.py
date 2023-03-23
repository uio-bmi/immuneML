from multiprocessing.pool import Pool
from pathlib import Path
import os
import numpy as np
import pandas as pd
import logging
import plotly.express as px
import warnings


from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.CompAIRRHelper import CompAIRRHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
import subprocess
import shutil


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
        self.cols_of_interest = ["organism", "antigen"] # todo make user setting
        self.label_name = None # todo make this optional to choose
        self.normalize_matches = True # todo make this optional

    @classmethod
    def build_object(cls, **kwargs):
        location = TCRMatchEpitopeAnalysis.__name__

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
        self.tcrmatch_files_path = PathBuilder.build(self.result_path / "tcrmatch_results_per_repertoire")

        tcrmatch_per_repertoire = self._run_tcrmatch_pipeline(self.dataset)

        return self._generate_report_results(tcrmatch_per_repertoire)

    def _generate_report_results(self, tcrmatch_per_repertoire):
        full_match_df = self._process_tcrmatch_output_files(tcrmatch_per_repertoire)
        self._annotate_repertoire_info(full_match_df, self.dataset)

        output_tables = [self._write_output_table(full_match_df, self.result_path / "tcrmatch_per_repertoire.tsv", name="Number of TCRMatch matches per repertoire")]
        summary_plots = []
        feature_plots = []

        for label_name in self.dataset.get_label_names():
            summarized_match_df = self._summarize_matches(full_match_df, label_name)

            output_tables.append(self._write_output_table(table=summarized_match_df,
                                                          file_path=self.result_path / f"label={label_name}_tcrmatch_summary.tsv",
                                                          name=f"Summary of TCRMatch results for different classes of label {label_name}"))

            classes = list(set(full_match_df[label_name]))

            if len(classes) == 2:
                summary_plots.append(self._safe_plot(plot_callable="_plot_binary_class_summary",
                                                    label_name=label_name,
                                                    classes=classes,
                                                    summary_df=summarized_match_df))

            feature_plots.extend(self._safe_plot(plot_callable="_plot_violin_per_feature",
                                                 match_df=full_match_df,
                                                 label_name=label_name))

        if len(self.dataset.get_label_names()) == 0:
            feature_plots.extend(self._safe_plot(plot_callable="_plot_violin_per_feature",
                                                 match_df=full_match_df,
                                                 label_name=None))

        return ReportResult(name=self.name,
                            info="TCRMatch matches per repertoire",
                            output_figures=[plot for plot in summary_plots + feature_plots if plot is not None],
                            output_tables=[table for table in output_tables if table is not None])

    def _run_tcrmatch_pipeline(self, dataset):
        with Pool(processes=self.number_of_processes) as pool:
            tcrmatch_files = pool.map(self._run_tcrmatch_pipeline_for_repertoire, dataset.get_data())

        return tcrmatch_files

    def _run_tcrmatch_pipeline_for_repertoire(self, repertoire: Repertoire):
        tcrmatch_infiles_for_rep_path = PathBuilder.build(self.result_path / f"tcrmatch_infiles_per_repertoire/{repertoire.identifier}")
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: made rep results path {tcrmatch_infiles_for_rep_path}")

        tcrmatch_input_files_path = PathBuilder.build(tcrmatch_infiles_for_rep_path / "tcrmatch_input_files")
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: made tcrmatch results path {tcrmatch_input_files_path}")

        repertoire_output_file_path = self.tcrmatch_files_path / f"{repertoire.identifier}.tsv"

        cdr3s_file = tcrmatch_infiles_for_rep_path / "cdr3_aas.txt"
        self._export_repertoire_cdr3s(cdr3s_file, repertoire)
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: rep cdr3s exported to {cdr3s_file}")

        pairs_file = self._create_pairs_file_with_compairr(tcrmatch_infiles_for_rep_path, cdr3s_file)
        logging.info(f"{TCRMatchEpitopeAnalysis.__name__}: pairs file at {pairs_file}")

        self._make_tcrmatch_input_files(pairs_file, tcrmatch_input_files_path)

        self._run_tcrmatch_on_each_file(tcrmatch_input_files_path, repertoire_output_file_path)

        if not self.keep_tmp_results:
            shutil.rmtree(tcrmatch_infiles_for_rep_path)

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


    def _process_tcrmatch_output_files(self, tcrmatch_files):
        dfs = []

        for file in tcrmatch_files:
            df = pd.read_csv(file, usecols=self.cols_of_interest, sep="\t")

            rep_df = df[self.cols_of_interest].value_counts().reset_index()

            rep_df.rename(columns={0: "repertoire_matches"}, inplace=True)
            rep_df["repertoire"] = file.stem

            dfs.append(rep_df)

        df = pd.concat(dfs)
        repertoire_names = set(df["repertoire"])

        df = pd.pivot(df, index=self.cols_of_interest, columns="repertoire", values="repertoire_matches").fillna(0)
        df.reset_index(inplace=True)
        df = pd.melt(df, id_vars=self.cols_of_interest, value_vars=repertoire_names, value_name="repertoire_matches")
        return df

    def _annotate_repertoire_info(self, df, dataset):
        self._annotate_repertoire_sizes(df, dataset)

        for label_name in dataset.get_label_names():
            self._annotate_repertoire_classes(df, dataset, label_name)

    def _annotate_repertoire_sizes(self, df, dataset):
        repertoire_sizes = {repertoire.identifier: repertoire.get_element_count() for repertoire in dataset.get_data()}
        df["repertoire_size"] = df["repertoire"].replace(repertoire_sizes)
        df["normalized_repertoire_matches"] = df["repertoire_matches"] / df["repertoire_size"]

    def _annotate_repertoire_classes(self, df, dataset, label_name):
        repertoire_metadata = dataset.get_metadata([label_name, "identifier"])
        repertoire_classes = {identifier: label for identifier, label in
                              zip(repertoire_metadata["identifier"], repertoire_metadata[label_name])}

        df[label_name] = df["repertoire"].replace(repertoire_classes)

    def _summarize_matches(self, df, label_name):
        summary_stats = ["mean", "min", "max"]
        value_column = "normalized_repertoire_matches" if self.normalize_matches else "repertoire_matches"
        classes = set(df[label_name])

        df = df.groupby(self.cols_of_interest + [label_name])[value_column].aggregate(summary_stats).reset_index()
        df = pd.pivot(df, index=self.cols_of_interest, columns=[label_name], values=summary_stats).reset_index()
        df.columns = ["_".join(col).rstrip("_") for col in df.columns.values]

        for label_class in classes:
            df[f"error_{label_class}_plus"] = df[f"max_{label_class}"] - df[f"mean_{label_class}"]
            df[f"error_{label_class}_minus"] = df[f"mean_{label_class}"] - df[f"min_{label_class}"]

        return df

    def _plot_violin_per_feature(self, match_df, label_name):
        y_col = "normalized_repertoire_matches" if self.normalize_matches else "repertoire_matches"
        y_col_name = "Normalized matches per repertoire" if self.normalize_matches else "Matches per repertoire"
        filename_prefix = f"label={label_name}_" if label_name is not None else ""
        report_output_name_suffix = f"label={label_name} and " if label_name is not None else ""

        report_outputs = []

        for features, feature_df in match_df.groupby(self.cols_of_interest):
            feature_strings = [f"{name}={value}" for name, value in zip(self.cols_of_interest, features)]

            fig = px.violin(feature_df, x=label_name, y=y_col, color=label_name, points='all', box=True,
                            color_discrete_sequence=self._get_color_discrete_sequence(feature_df, label_name),
                            labels={y_col: y_col_name}, title="<br>".join(feature_strings),
                            hover_data=["repertoire", "repertoire_matches",
                                        "normalized_repertoire_matches", "repertoire_size"])
            fig.update_traces(meanline_visible=True)

            figure_path = str(self.result_path / f"{filename_prefix}{'_'.join(feature_strings)}.html")
            fig.write_html(figure_path)

            report_outputs.append(ReportOutput(path=Path(figure_path),
                                               name=f"Matches per repertoire for {report_output_name_suffix}{', '.join(feature_strings)}"))

        return report_outputs

    def _get_color_discrete_sequence(self, feature_df, label_name):
        if label_name is not None and label_name in feature_df:
            if len(set(feature_df[label_name])) == 2:
                return [px.colors.diverging.Tealrose[0], px.colors.diverging.Tealrose[-1]]

        return px.colors.diverging.Tealrose

    def _plot_binary_class_summary(self, summary_df, label_name, classes):
        matches_str = "Normalized matches" if self.normalize_matches else "Matches"

        max_axis = max(max(summary_df[f"mean_{classes[0]}"] + summary_df[f"error_{classes[0]}_plus"]),
                       max(summary_df[f"mean_{classes[1]}"] + summary_df[f"error_{classes[1]}_plus"]))

        fig = px.scatter(summary_df, y=f"mean_{classes[0]}", x=f"mean_{classes[1]}",
                         error_y=f"error_{classes[0]}_plus", error_y_minus=f"error_{classes[0]}_minus",
                         error_x=f"error_{classes[1]}_plus", error_x_minus=f"error_{classes[1]}_minus",
                         template='plotly_white',
                         range_x=[0, max_axis], range_y=[0, max_axis],
                         labels={f"mean_{classes[0]}": f"{matches_str} for {label_name}={classes[0]} repertoires",
                                 f"mean_{classes[1]}": f"{matches_str} for {label_name}={classes[1]} repertoires"},
                         hover_data=self.cols_of_interest)

        fig.update_traces(marker_color="rgb(0, 147, 146)",
                          error_x_color="rgba(114, 170, 161, 0.4)",
                          error_y_color="rgba(114, 170, 161, 0.4)")

        # add diagonal
        fig.update_layout(shapes=[{'type': "line", 'line': dict(color="#B0C2C7", dash="dash"),
                                   'yref': 'paper', 'xref': 'paper',
                                   'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1, 'layer': 'below'}])

        figure_path = str(self.result_path / f"tcrmatch_summary_label={label_name}_{classes[0]}_vs_{classes[1]}.html")
        fig.write_html(figure_path)

        return ReportOutput(path=Path(figure_path), name=f"TCRMatch summary label={label_name}, {classes[0]} vs {classes[1]}")

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            warnings.warn(f"{TCRMatchEpitopeAnalysis.__name__}: report can be generated only for a RepertoireDataset. Skipping this report...")
            return False
