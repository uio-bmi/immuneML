import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from immuneML.data_model.datasets.Dataset import Dataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ReportUtil import ReportUtil
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class VJGeneDistribution(DataReport):
    """
    This report creates several plots to gain insight into the V and J gene distribution of a given dataset.
    When a label is provided, the information in the plots is separated per label value, either by color or by creating
    separate plots. This way one can for example see if a particular V or J gene is more prevalent across disease
    associated receptors.

    - Individual V and J gene distributions: for sequence and receptor datasets, a bar plot is created showing how often
      each V or J gene occurs in the dataset. For repertoire datasets, boxplots are used to represent how often each V or J
      gene is used across all repertoires. Since repertoires may differ in size, these counts are normalised by the repertoire
      size (original count values are additionaly exported in tsv files).

    - Combined V and J gene distributions: for sequence and receptor datasets, a heatmap is created showing how often each
      combination of V and J genes occurs in the dataset. A similar plot is created for repertoire datasets, except in this
      case only the average value for the normalised gene usage frequencies are shown (original count values are additionaly exported in tsv files).


    **Specification arguments:**

    - split_by_label (bool): Whether to split the plots by a label. If set to true, the Dataset must either contain a single label, or alternatively the label of interest can be specified under 'label'. By default, split_by_label is False.

    - label (str): Optional label for separating the results by color/creating separate plots. Note that this should the name of a valid dataset label.

    - is_sequence_label (bool): for RepertoireDatasets, indicates if the label applies to the sequence level
      (e.g., antigen binding versus non-binding across repertoires) or repertoire level (e.g., diseased repertoires versus healthy repertoires).
      By default, is_sequence_label is False. For Sequence- and ReceptorDatasets, this parameter is ignored.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_vj_gene_report:
                    VJGeneDistribution:
                        label: ag_binding

    """

    @classmethod
    def build_object(cls, **kwargs):
        location = VJGeneDistribution.__name__

        ReportUtil.update_split_by_label_kwargs(kwargs, location)

        return VJGeneDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1,
                 name: str = None, split_by_label: bool = None, label: str = None, is_sequence_label: bool = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.split_by_label = split_by_label
        self.label_name = label
        self.is_sequence_label = is_sequence_label

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self._set_label_name()

        if isinstance(self.dataset, SequenceDataset) or isinstance(self.dataset, ReceptorDataset):
            report_result = self._get_sequence_receptor_results()

        else:
            report_result = self._get_repertoire_results()

        return report_result

    def _set_label_name(self):
        if self.split_by_label:
            if self.label_name is None:
                self.label_name = list(self.dataset.get_label_names())[0]
        else:
            self.label_name = None

    def _get_sequence_receptor_results(self):
        attributes = [self.label_name] if self.split_by_label else []
        attributes += ["v_call", "j_call", "locus"]
        dataset_attributes = self.dataset.data.topandas()[attributes]
        dataset_attributes = {key: value.tolist() for key, value in dataset_attributes.items()}

        v_tables, v_plots = self._get_single_gene_results_from_attributes(dataset_attributes, "v_call")
        j_tables, j_plots = self._get_single_gene_results_from_attributes(dataset_attributes, "j_call")
        vj_tables, vj_plots = self._get_combo_gene_results_from_attributes(dataset_attributes)

        return ReportResult(name=self.name,
                            info="V and J gene distributions",
                            output_figures=v_plots + j_plots + vj_plots,
                            output_tables=v_tables + j_tables + vj_tables)

    def _get_single_gene_results_from_attributes(self, dataset_attributes, call_type):
        vj = call_type[0].upper()

        tables = []
        plots = []

        counts_df = self._get_gene_count_df(dataset_attributes, call_type)

        for chain in set(dataset_attributes["locus"]):
            chain_df = counts_df[counts_df["locus"] == chain]

            tables.append(self._write_output_table(chain_df,
                                                   file_path=self.result_path / f"{chain}{vj}_gene_distribution.tsv",
                                                   name=f"{vj} gene distribution"))

            plots.append(self._safe_plot(plot_callable="_plot_gene_distribution",
                                         df=chain_df,
                                         title=f"{chain} {vj} gene distribution",
                                         filename=f"{chain}{vj}_gene_distribution.html"))

        return tables, plots

    def _get_gene_count_df(self, dataset_attributes, call_type, include_label=True):
        if self.label_name is None or include_label == False:
            genes_counter = Counter(zip(dataset_attributes[call_type],
                                        dataset_attributes["locus"]))
            colnames = ["genes", "locus"]
        else:
            genes_counter = Counter(zip(dataset_attributes[call_type],
                                        dataset_attributes["locus"],
                                        dataset_attributes[self.label_name]))
            colnames = ["genes", "locus", self.label_name]

        return self._counter_to_df(genes_counter, colnames)

    def _get_vj_combo_count_df(self, dataset_attributes, include_label=True):
        if self.label_name is None or include_label == False:
            genes_counter = Counter(zip(dataset_attributes["v_call"],
                                        dataset_attributes["j_call"],
                                        dataset_attributes["locus"]))
            colnames = ["v_genes", "j_genes", "locus"]
        else:
            genes_counter = Counter(zip(dataset_attributes["v_call"],
                                        dataset_attributes["j_call"],
                                        dataset_attributes["locus"],
                                        dataset_attributes[self.label_name]))
            colnames = ["v_genes", "j_genes", "locus", self.label_name]

        return self._counter_to_df(genes_counter, colnames)

    def _counter_to_df(self, counter, colnames):
        df = pd.DataFrame.from_dict(counter, orient="index", columns=["counts"])
        df[colnames] = pd.DataFrame(df.index.tolist(), index=df.index)

        return df[colnames + ["counts"]].reset_index(drop=True)

    def _plot_gene_distribution(self, df, title, filename):
        figure = px.bar(df, x="genes", y="counts", color=self.label_name,
                        labels={"genes": "Gene names",
                                "counts": "Observed frequency"},
                        color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(xaxis=dict(tickmode='array', tickvals=df["genes"]),
                             yaxis=dict(tickmode='array', tickvals=df["counts"]),
                             template="plotly_white",
                             barmode="group")

        file_path = self.result_path / filename
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=title)

    def _get_combo_gene_results_from_attributes(self, dataset_attributes):
        tables = []
        plots = []

        vj_combo_count_df = self._get_vj_combo_count_df(dataset_attributes)


        for chain in set(dataset_attributes["locus"]):
            chain_df = vj_combo_count_df[vj_combo_count_df["locus"] == chain]

            tables.append(self._write_output_table(chain_df,
                                                   file_path=self.result_path / f"{chain}VJ_gene_distribution.tsv",
                                                   name=f"Combined {chain} V+J gene distribution"))


            if not self.split_by_label:
                plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                             chain_df=chain_df,
                                             title=f"Combined {chain} V+J gene distribution",
                                             filename=f"{chain}VJ_gene_distribution.html"))
            else:
                # ensure the same color scale is used for each heatmap
                zmax = max(chain_df["counts"])

                for label_class in set(dataset_attributes[self.label_name]):
                    label_chain_df = chain_df[chain_df[self.label_name] == label_class]

                    plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                                 chain_df=label_chain_df,
                                                 title=f"Combined {chain} V+J gene distribution for {self.label_name}={label_class}",
                                                 filename=f"{chain}VJ_gene_distribution_{self.label_name}={label_class}.html",
                                                 zmax=zmax))

        return tables, plots

    def _plot_gene_combo_heatmap(self, chain_df, title, filename, value_to_plot="counts", zmax=None,
                                 color_name="Observed frequency"):
        zmax = max(chain_df[value_to_plot]) if zmax is None else zmax

        chain_df = chain_df.pivot(index="v_genes", columns="j_genes", values=value_to_plot).round(decimals=2)
        figure = px.imshow(chain_df, labels=dict(x="V genes", y="J genes", color=color_name),
                           text_auto=True, zmin=0, zmax=zmax, aspect="auto")

        figure.update_traces(hoverongaps=False)
        figure.update_layout(template='plotly_white', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

        file_path = self.result_path / filename
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=title)

    def _get_repertoire_results(self):
        v_df, j_df, vj_df = self._get_repertoire_count_dfs()

        tables = self._write_repertoire_tables(v_df, j_df, vj_df)

        plots = []

        for chain in set(vj_df["locus"]):
            plots.append(self._safe_plot(plot_callable="_plot_gene_distribution_across_repertoires",
                                         chain_df=v_df[v_df["locus"] == chain],
                                         title=f"{chain} V gene distribution per repertoire",
                                         filename=f"{chain}V_gene_distribution.html"))

            plots.append(self._safe_plot(plot_callable="_plot_gene_distribution_across_repertoires",
                                         chain_df=j_df[j_df["locus"] == chain],
                                         title=f"{chain} J gene distribution per repertoire",
                                         filename=f"{chain}J_gene_distribution.html"))

            mean_chain_vj_df = self._average_norm_counts_per_repertoire(chain_vj_df=vj_df[vj_df["locus"] == chain])

            tables.append(self._write_output_table(mean_chain_vj_df,
                                                   file_path=self.result_path / f"{chain}VJ_gene_distribution_averaged_across_repertoires.tsv",
                                                   name=f"Combined {chain} V+J gene distribution averaged across repertoires"))

            plots.extend(self._get_repertoire_heatmaps(mean_chain_vj_df, chain))

        return ReportResult(name=self.name,
                            info="V and J gene distributions per repertoire",
                            output_figures=plots,
                            output_tables=tables)

    def _get_repertoire_heatmaps(self, mean_chain_vj_df, chain):
        plots = []
        if not self.split_by_label:
            plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                         chain_df=mean_chain_vj_df,
                                         title=f"Combined {chain} V+J gene distribution averaged across repertoires",
                                         filename=f"{chain}VJ_gene_distribution_averaged_across_repertoires.html",
                                         value_to_plot="mean_norm_counts",
                                         color_name="Average observed frequency<br>across repertoires<br>normalised by repertoire size"))
        else:
            for label_class in set(mean_chain_vj_df[self.label_name]):
                plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                             chain_df=mean_chain_vj_df[
                                                 mean_chain_vj_df[self.label_name] == label_class],
                                             title=f"Combined {chain} V+J gene distribution for {self.label_name}={label_class} averaged across repertoires",
                                             filename=f"{chain}VJ_gene_distribution_{self.label_name}={label_class}_averaged_across_repertoires.html",
                                             zmax=max(mean_chain_vj_df["mean_norm_counts"]),
                                             value_to_plot="mean_norm_counts",
                                             color_name="Average observed frequency<br>across repertoires<br>normalised by repertoire size"))

        return plots

    def _average_norm_counts_per_repertoire(self, chain_vj_df):
        groupby_cols = ["v_genes", "j_genes", "locus"]
        groupby_cols += [self.label_name] if self.label_name is not None else []

        mean_chain_vj_df = pd.DataFrame(chain_vj_df.groupby(groupby_cols)["norm_counts"].mean()).reset_index()
        mean_chain_vj_df.rename(columns={"norm_counts": "mean_norm_counts"}, inplace=True)

        return mean_chain_vj_df

    def _get_repertoire_count_dfs(self):
        v_dfs = []
        j_dfs = []
        vj_dfs = []

        for repertoire in self.dataset.repertoires:
            data = repertoire.data
            if hasattr(data, "locus"):
                assert len(set(data.locus.tolist())) == 1, (f"{VJGeneDistribution.__name__}: Repertoire {repertoire.identifier} of dataset {self.dataset.name} contained multiple loci: {set(data.locus.tolist())}. "
                                                              f"This report can only be created for 1 locus per repertoire.")


            repertoire_attributes = {"v_call": data.v_call.tolist(),
                                     "j_call": data.j_call.tolist(),
                                     "locus": data.locus.tolist()}

            if self.is_sequence_label:
                repertoire_attributes[self.label_name] = getattr(data, self.label_name).tolist()

            v_rep_df = self._get_gene_count_df(repertoire_attributes, "v_call", include_label=self.is_sequence_label)
            j_rep_df = self._get_gene_count_df(repertoire_attributes, "j_call", include_label=self.is_sequence_label)
            vj_rep_df = self._get_vj_combo_count_df(repertoire_attributes, include_label=self.is_sequence_label)

            self._supplement_repertoire_df(v_rep_df, repertoire)
            self._supplement_repertoire_df(j_rep_df, repertoire)
            self._supplement_repertoire_df(vj_rep_df, repertoire)

            v_dfs.append(v_rep_df)
            j_dfs.append(j_rep_df)
            vj_dfs.append(vj_rep_df)

        return pd.concat(v_dfs, ignore_index=True), \
            pd.concat(j_dfs, ignore_index=True), \
            pd.concat(vj_dfs, ignore_index=True),

    def _supplement_repertoire_df(self, rep_df, repertoire):
        rep_df["repertoire_id"] = repertoire.identifier
        rep_df["subject_id"] = repertoire.metadata["subject_id"] if "subject_id" in repertoire.metadata else ''
        rep_df["repertoire_size"] = repertoire.get_element_count()
        rep_df["norm_counts"] = rep_df["counts"] / rep_df["repertoire_size"]

        if not self.is_sequence_label and self.label_name is not None:
            rep_df[self.label_name] = repertoire.metadata[self.label_name]

    def _write_repertoire_tables(self, v_df, j_df, vj_df):
        tables = []

        for chain in set(v_df["locus"]):
            tables.append(self._write_output_table(v_df[v_df['locus'] == chain],
                                                   file_path=self.result_path / f"{chain}V_gene_distribution.tsv",
                                                   name=f"{chain}V gene distribution per repertoire"))

            tables.append(self._write_output_table(j_df[j_df['locus'] == chain],
                                                   file_path=self.result_path / f"{chain}J_gene_distribution.tsv",
                                                   name=f"{chain}J gene distribution per repertoire"))

            tables.append(self._write_output_table(vj_df[vj_df['locus'] == chain],
                                                   file_path=self.result_path / f"{chain}VJ_gene_distribution.tsv",
                                                   name=f"Combined {chain}V+J gene distribution"))

        return tables

    def _plot_gene_distribution_across_repertoires(self, chain_df, title, filename):
        figure = px.box(chain_df, x="genes", y="norm_counts", color=self.label_name,
                        hover_data=["repertoire_id", "subject_id"],
                        labels={"genes": "Gene names",
                                "norm_counts": "Fraction of the repertoire"},
                        color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(template="plotly_white")

        file_path = self.result_path / filename
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=title)

    def check_prerequisites(self):
        if self.split_by_label:
            if self.label_name is None:
                if len(self.dataset.get_label_names()) != 1:
                    logging.warning(
                        f"{VJGeneDistribution.__name__}: ambiguous label: split_by_label was set to True but no label "
                        f"name was specified, and the number of available labels is "
                        f"{len(self.dataset.get_label_names())}: {self.dataset.get_label_names()}. Skipping this "
                        f"report...")
                    return False
            else:
                if not self.is_sequence_label and self.label_name not in self.dataset.get_label_names():
                    logging.warning(f"{VJGeneDistribution.__name__}: the specified label name ({self.label_name}) was not available among the dataset labels: {self.dataset.get_label_names()}. If this is a sequence label, please set is_sequence_label to True. Skipping this report...")
                    return False

        if isinstance(self.dataset, ReceptorDataset) or isinstance(self.dataset, SequenceDataset):
            try:
                self.dataset.data.topandas()[["v_call", "j_call", "locus"]]
            except AttributeError as e:
                logging.warning(
                    f"{VJGeneDistribution.__name__}: the following attributes were expected to be present in the "
                    f"dataset: v_call, j_call, locus. The following error occurred when attempting to get these "
                    f"attributes: {e} Skipping this report...")
                return False

        return True
