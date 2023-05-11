import logging
import math
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import regex as re

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.simulation.SequenceDispenser import SequenceDispenser
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.signal_implanting_strategy.DecoyImplanting import DecoyImplanting
from immuneML.util import Logger
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class SequenceGenerationProbabilityDistribution(DataReport):
    """
    Generates a plot of the distribution of generation probability and number of occurrences of the sequences in a RepertoireDataset.

    Arguments:
        count_by_repertoire (bool): Whether to have the appearance rate of a sequence be decided by how many repertoires it
        appears in (True) or by the total sequence count in all repertoires (False).
        Default value is False.
        mark_implanted_labels (bool): Plot the implanted sequences with different colors. Default value is True.
        default_sequence_label (str): Name of main dataset (all non-implanted sequences) in the plot legend. Default value is "dataset".

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_p_gen_report:
            SequenceGenerationProbabilityDistribution:
                count_by_repertoire: False
                mark_implanted_labels: True
                default_sequence_label: OLGA
    """

    @classmethod
    def build_object(cls,
                     **kwargs):  # called when parsing YAML - all checks for parameters (if any) should be in this function
        return SequenceGenerationProbabilityDistribution(**kwargs)

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None,
                 name: str = None, count_by_repertoire: bool = False, mark_implanted_labels: bool = True,
                 default_sequence_label: str = "dataset"):
        super().__init__(dataset=dataset, result_path=result_path, name=name)

        ParameterValidator.assert_type_and_value(count_by_repertoire, bool,
                                                 SequenceGenerationProbabilityDistribution.__name__,
                                                 "count_by_repertoire")
        self.count_by_repertoire = count_by_repertoire

        ParameterValidator.assert_type_and_value(mark_implanted_labels, bool,
                                                 SequenceGenerationProbabilityDistribution.__name__,
                                                 "mark_implanted_labels")
        self.mark_implanted_labels = mark_implanted_labels

        ParameterValidator.assert_type_and_value(default_sequence_label, str,
                                                 SequenceGenerationProbabilityDistribution.__name__,
                                                 "default_sequence_label")
        self.default_sequence_label = default_sequence_label

    def check_prerequisites(
            self):  # called at runtime to check if the report can be run with params assigned at runtime (e.g., dataset is set at runtime)
        if isinstance(self.dataset, RepertoireDataset):
            return True
        else:
            logging.warning("Report can be generated only from RepertoireDataset. Skipping this report...")
            return False

    def _generate(self) -> ReportResult:  # the function that creates the report

        dataset_df = self._load_dataset_dataframe()

        dataset_df = self._get_sequence_count(dataset_df)
        dataset_df = self._get_sequence_pgen(dataset_df)

        report_output_fig = self._safe_plot(dataset_df=dataset_df)
        output_figures = None if report_output_fig is None else [report_output_fig]

        dataset_df.to_csv(self.result_path / f"processed_dataset_with_pgen.csv", index=False)
        dataset_output = ReportOutput(self.result_path / f"processed_dataset_with_pgen.csv",
                                      "Processed dataset with generation probability and implanted labels for each sequence")
        output_tables = None if dataset_df is None else [dataset_output,
                                                         self._generate_occurrence_limit_pgen_range(dataset_df),
                                                         *self._create_output_table_for_vdjRec(dataset_df,
                                                                                               self._load_dataset_dataframe())
                                                         ]

        Logger.print_log(
            f"Finished report",
            include_datetime=True)

        return ReportResult(type(self).__name__, output_figures=output_figures, output_tables=output_tables)

    def _get_sequence_pgen(self, dataset_df) -> pd.DataFrame:
        """
        Computes generation probability of each sequence from self.dataset.

        Returns:
            np.ndarray: Generation probabilities of sequences
        """

        olga = OLGA.build_object(model_path=None,
                                 default_model_name=SequenceDispenser.get_default_model_name(self.dataset),
                                 chain=None, use_only_productive=False)
        olga.load_model()
        self.olga = olga

        Logger.print_log(
            f"Starting generation probability-calculation using {self.number_of_processes} threads ({dataset_df.sequence_aas.unique().size} unique sequences)",
            include_datetime=True)

        dfs = np.array_split(dataset_df, self.number_of_processes)
        pool = mp.Pool(self.number_of_processes)
        results = pool.map(self._compute_pgen, [df for df in dfs])

        pool.close()

        return pd.concat(results)

    def _compute_pgen(self, dataframe, sequence_type=SequenceType.AMINO_ACID):
        Logger.print_log("Starting pgen computation in a thread", include_datetime=True)
        dataframe["pgen"] = self.olga.compute_p_gens(dataframe, sequence_type)
        Logger.print_log("Finished pgen computation in a thread", include_datetime=True)
        return dataframe

    def _get_sequence_count(self, dataset_df) -> pd.DataFrame:
        """
        Either counts number of duplicates of each sequence from dataset, or how many repertoires each sequence
        appears in. This is specified in yaml file.

        Returns:
            dict: Dictionary where key is amino acid sequence (str) and value is sequence count (int)
        """

        # count by repertoire occurrences: drop duplicates with same sequence, v/j-gene and repertoire before count
        if self.count_by_repertoire:
            dataset_df.drop_duplicates(inplace=True, subset=["sequence_aas", "v_genes", "j_genes", "repertoire"])

        sequence_columns = ["sequence_aas", "v_genes", "j_genes"]
        if self.count_by_repertoire:
            count_df = dataset_df.groupby(sequence_columns)["repertoire"].count().reset_index(name="count")
            dataset_with_count = pd.merge(dataset_df, count_df, how="inner", on=sequence_columns).drop_duplicates(
                ignore_index=True, subset=sequence_columns)
        else:

            count = defaultdict(int)

            for row in dataset_df.itertuples():
                count[row.sequence_aas] += row.duplicate_count

            count_df = pd.DataFrame(data={"sequence_aas": count.keys(), "count": count.values()})
            dataset_with_count = pd.merge(dataset_df, count_df, how="inner", on="sequence_aas").drop_duplicates(
                ignore_index=True, subset=sequence_columns)

        dataset_with_count.drop(columns=["duplicate_count"])

        return dataset_with_count

    def _plot(self, dataset_df) -> ReportOutput:

        Logger.print_log(
            f"Starting plot generation",
            include_datetime=True)

        # turn pgen into log10
        df_log = dataset_df.copy()
        df_log = df_log.loc[df_log["pgen"] != 0]
        df_log["pgen"] = df_log["pgen"].apply(lambda x: math.log10(x))

        figure = go.Figure()

        # separate into violin and scatter by label
        violin_df = df_log.loc[df_log["label"] == self.default_sequence_label]
        figure.add_trace(
            go.Violin(x=violin_df["count"], y=violin_df["pgen"],
                      name=self.default_sequence_label, points=False, bandwidth=0.05))

        line = violin_df.groupby(by=["count"])["pgen"].min().to_dict()
        figure.add_trace(go.Scatter(x=list(line.keys()), y=list(line.values()),
                                    name=self.default_sequence_label + " (lowest P<sub>gen</sub>)", mode="lines",
                                    line_color="#a6c1ff"))

        signal_df = df_log.loc[df_log["label"] != self.default_sequence_label]
        for signal in signal_df["label"].unique():
            label_df = signal_df.loc[df_log["label"] == signal]
            figure.add_trace(go.Scatter(x=label_df["count"], y=label_df["pgen"], mode="markers", name=signal))

        """if self.mark_implanted_labels:
            figure = px.strip(df_log, x="count", y="pgen", hover_data=["sequence_aas"], color="label",
                              stripmode="overlay")

        else:
            figure = px.strip(df_log, x="count", y="pgen", hover_data=["sequence_aas"])    
        figure.update_traces(jitter=1.0)
        """

        xaxis_title = "Clone occurrences" if self.count_by_repertoire else "Count occurrences"
        figure.update_layout(title="Sequence generation probability distribution", template="plotly_white",
                             xaxis=dict(tickmode="linear",
                                        tick0=0,
                                        dtick=max(1, round(max(df_log["count"]) / 300) * 10)),
                             xaxis_title=xaxis_title,
                             yaxis_title="log<sub>10</sub> P<sub>gen</sub>")
        figure.layout.legend.itemsizing = "constant"

        Logger.print_log(
            f"Exporting plot",
            include_datetime=True)

        PathBuilder.build(self.result_path)
        file_path = self.result_path / "pgen_distribution_plot.html"
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name="sequence generation probability distribution plot")

    def _load_dataset_dataframe(self) -> pd.DataFrame:

        Logger.print_log(
            f"Loading dataset",
            include_datetime=True)

        dfs = []

        label_names = []
        if self.mark_implanted_labels:
            label_names = list(self.dataset.get_label_names()) + [DecoyImplanting.__name__]

        for repertoire in self.dataset.get_data():

            rep_dict = {
                "sequence_aas": repertoire.get_sequence_aas(),
                "v_genes": repertoire.get_v_genes(),
                "j_genes": repertoire.get_j_genes(),
                "repertoire": repertoire.identifier,
                "duplicate_count": repertoire.get_counts(),
            }

            for label in label_names:
                rep_dict[label] = repertoire.get_attribute(label)

            rep_df = pd.DataFrame(data=rep_dict)

            dfs.append(
                rep_df
            )

        full_dataset = pd.concat(dfs, ignore_index=True)

        if label_names:
            full_dataset["label"] = full_dataset[label_names].apply(
                lambda x: "".join([re.findall("signal_id='([^\"']+)", i)[0] for i in x if i]),
                axis=1).replace([None, ""], self.default_sequence_label)
        else:
            full_dataset["label"] = self.default_sequence_label

        full_dataset.drop(label_names, axis=1, inplace=True)

        # sum duplicate_count for identical rows
        full_dataset = full_dataset.groupby(["sequence_aas", "v_genes", "j_genes", "repertoire", "label"]).sum()
        full_dataset.reset_index(inplace=True)

        return full_dataset

    def _create_output_table_for_vdjRec(self, pgen_df, full_df):

        Logger.print_log(
            f"Creating vdjRec output tables",
            include_datetime=True)

        path = self.result_path
        name = "pgen_dataset_for_hacking"

        df = pd.merge(full_df, pgen_df[["sequence_aas", "v_genes", "j_genes", "pgen", "count"]], how="inner",
                      on=["sequence_aas", "v_genes", "j_genes"])

        df = df.loc[df["count"] > 1]

        if self.mark_implanted_labels:
            target_names = list(self.dataset.get_label_names())
        else:
            target_names = []

        Logger.print_log(
            f"START SEQUENCE COUNT FILE",
            include_datetime=True)
        # MAKE SEQUENCE COUNT FILE:
        repertoires = full_df["repertoire"].unique()

        sequence_df = pd.DataFrame()

        sequence_df["CDR3.amino.acid.sequence"] = df["sequence_aas"]
        # TODO find out what sim_num is!
        sequence_df["sim_num"] = 0

        for r in repertoires:
            sequence_df[r] = 0

        # TODO parallelize
        for row in df.itertuples():
            sequence_df.at[row.Index, row.repertoire] += 1

        sequence_df["Sum"] = df["count"]
        sequence_df["target"] = ["TRUE" if label in target_names else "FALSE" for label in df["label"]]
        sequence_df["Pgen"] = df["pgen"]

        sequence_df = pd.merge(sequence_df.groupby(["CDR3.amino.acid.sequence"])[repertoires].sum(),
                               sequence_df[["sim_num", "CDR3.amino.acid.sequence", "Sum", "target", "Pgen"]],
                               how="inner", on="CDR3.amino.acid.sequence").drop_duplicates().reset_index(drop=True)

        # move sim_num to front
        cols = sequence_df.columns.tolist()
        cols = [cols[-4]] + cols[:-4] + cols[-3:]
        sequence_df = sequence_df[cols]

        sequence_df.to_csv(path / f"sequences_{name}.csv", sep=";")

        Logger.print_log(
            f"START SAMPLES FILE",
            include_datetime=True)

        # MAKE SAMPLES FILE:
        samples_df = pd.DataFrame()

        full_df = full_df[full_df["sequence_aas"].isin(df["sequence_aas"])]

        samples_df["count"] = full_df.groupby(["repertoire"])["repertoire"].count()
        samples_df.reset_index(inplace=True)
        samples_df["analysis"] = "TRUE"
        samples_df.rename(columns={"repertoire": "sample"}, inplace=True)

        samples_df.to_csv(path / f"samples_{name}.csv", sep=";")

        return [
            ReportOutput(path / f"sequences_{name}.csv", "Dataset with pgen and target for hacking method: SEQUENCES"),
            ReportOutput(path / f"samples_{name}.csv", "Dataset with pgen and target for hacking method: SAMPLES")]

    def _generate_occurrence_limit_pgen_range(self, dataset_df):
        """
        Generates occurrence limit pgen range for MutatedSequenceImplanting

        Example:
        occurrence_limit_pgen_range:
                1e-10: 2
                1e-7: 3
                1e-6: 5
        """

        Logger.print_log(
            f"Creating occurrence limit pgen range",
            include_datetime=True)

        path = self.result_path / "occurrence_limit_pgen_range.csv"

        non_implanted_df = dataset_df.loc[dataset_df["label"] == self.default_sequence_label]
        occurrence_limit_df = non_implanted_df.loc[non_implanted_df["pgen"] > 0][non_implanted_df["count"] > 1].groupby(["count"])[
            "pgen"].min().to_frame()

        lowest_pgen = None
        for count, row in occurrence_limit_df.iloc[::-1].iterrows():
            if lowest_pgen is None or row["pgen"] < lowest_pgen:
                lowest_pgen = row["pgen"]
            else:
                occurrence_limit_df.drop(count, inplace=True)

        f = open(path, "w")
        f.write("occurrence_limit_pgen_range:\n")

        for count, row in occurrence_limit_df.iterrows():
            f.write(f"  {row['pgen']}: {count}\n")
        f.close()

        return ReportOutput(path, "Occurrence limit pgen range for MutatedSequenceImplanting")
