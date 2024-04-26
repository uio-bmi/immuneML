import warnings
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import numpy as np

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
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

        if kwargs["label"] is not None:
            ParameterValidator.assert_type_and_value(kwargs["label"], str, location, "label")

            if kwargs["split_by_label"] is False:
                warnings.warn(f"{location}: label is set but split_by_label was False, setting split_by_label to True")
                kwargs["split_by_label"] = True

        return VJGeneDistribution(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None,
                 split_by_label: bool = None, label: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)
        self.split_by_label = split_by_label
        self.label_name = label

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        self._set_label_name()

        if isinstance(self.dataset, SequenceDataset) or isinstance(self.dataset, ReceptorDataset):
            report_result = self._get_sequence_receptor_results()

        elif isinstance(self.dataset, RepertoireDataset):
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
        attributes += ["v_call", "j_call", "chain"]
        dataset_attributes = self.dataset.get_attributes(attributes=attributes,
                                                         as_list=False)
        dataset_attributes = {key: value.tolist() for key, value in dataset_attributes.items()}

        v_tables, v_plots = self._get_single_gene_results_from_attributes(dataset_attributes, "v_call")
        j_tables, j_plots = self._get_single_gene_results_from_attributes(dataset_attributes, "j_call")
        vj_tables, vj_plots = self._get_combo_gene_results_from_attributes(dataset_attributes)

        return ReportResult(name=self.name,
                            info="V and J gene distributions",
                            output_figures=v_plots+j_plots+vj_plots,
                            output_tables=v_tables+j_tables+vj_tables)

    def _get_single_gene_results_from_attributes(self, dataset_attributes, call_type):
        vj = call_type[0].upper()

        tables = []
        plots = []

        counts_df = self._get_gene_count_df(dataset_attributes, call_type)
        tables.append(self._write_output_table(counts_df,
                                               file_path=self.result_path / f"{vj}_gene_distribution.tsv",
                                               name=f"{vj} gene distribution"))

        for chain in set(dataset_attributes["chain"]):
            plots.append(self._safe_plot(plot_callable="_plot_gene_distribution",
                                         df=counts_df[counts_df["chain"] == chain],
                                         title=f"{chain} {vj} gene distribution",
                                         filename=f"{chain}_{vj}_gene_distribution.html"))

        return tables, plots

    def _get_gene_count_df(self, dataset_attributes, call_type, include_label=True):
        if self.label_name is None or include_label==False:
            genes_counter = Counter(zip(dataset_attributes[call_type],
                                        dataset_attributes["chain"]))
            colnames = ["genes", "chain"]
        else:
            genes_counter = Counter(zip(dataset_attributes[call_type],
                                        dataset_attributes["chain"],
                                        dataset_attributes[self.label_name]))
            colnames = ["genes", "chain", self.label_name]

        return self._counter_to_df(genes_counter, colnames)

    def _get_vj_combo_count_df(self, dataset_attributes, include_label=True):
        if self.label_name is None or include_label==False:
            genes_counter = Counter(zip(dataset_attributes["v_call"],
                                        dataset_attributes["j_call"],
                                        dataset_attributes["chain"]))
            colnames = ["v_genes", "j_genes", "chain"]
        else:
            genes_counter = Counter(zip(dataset_attributes["v_call"],
                                        dataset_attributes["j_call"],
                                        dataset_attributes["chain"],
                                        dataset_attributes[self.label_name]))
            colnames = ["v_genes", "j_genes", "chain", self.label_name]

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
        tables.append(self._write_output_table(vj_combo_count_df,
                                               file_path=self.result_path / f"VJ_gene_distribution.tsv",
                                               name=f"Combined V+J gene distribution"))

        for chain in set(dataset_attributes["chain"]):
            chain_df = vj_combo_count_df[vj_combo_count_df["chain"] == chain]

            if not self.split_by_label:
                plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                             chain_df=chain_df,
                                             title=f"Combined {chain} V+J gene distribution",
                                             filename=f"{chain}_VJ_gene_distribution.html"))
            else:
                # ensure the same color scale is used for each heatmap
                zmax = max(chain_df["counts"])

                for label_class in set(dataset_attributes[self.label_name]):
                    label_chain_df = chain_df[chain_df[self.label_name] == label_class]

                    plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                                 chain_df=label_chain_df,
                                                 title=f"Combined {chain} V+J gene distribution for {self.label_name}={label_class}",
                                                 filename=f"{chain}_VJ_gene_distribution_{self.label_name}={label_class}.html",
                                                 zmax=zmax))

        return tables, plots

    def _plot_gene_combo_heatmap(self, chain_df, title, filename, value_to_plot="counts", zmax=None, color_name="Observed frequency"):
        zmax = max(chain_df[value_to_plot]) if zmax is None else zmax

        chain_df = chain_df.pivot(index="v_genes", columns="j_genes", values=value_to_plot).round(decimals=2)
        figure = px.imshow(chain_df, labels=dict(x="V genes",
                                                 y="J genes",
                                                 color=color_name),
                           text_auto=True, zmin=0, zmax=zmax,
                           aspect="auto")

        file_path = self.result_path / filename
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=title)


    def _get_repertoire_results(self):
        v_df, j_df, vj_df = self._get_repertoire_count_dfs()

        tables = self._write_repertoire_tables(v_df, j_df, vj_df)

        plots = []

        for chain in set(vj_df["chain"]):
            plots.append(self._safe_plot(plot_callable="_plot_gene_distribution_across_repertoires",
                                         chain_df=v_df[v_df["chain"]==chain],
                                         title=f"{chain} V gene distribution per repertoire",
                                         filename=f"{chain}_V_gene_distribution.html"))

            plots.append(self._safe_plot(plot_callable="_plot_gene_distribution_across_repertoires",
                                         chain_df=j_df[j_df["chain"]==chain],
                                         title=f"{chain} J gene distribution per repertoire",
                                         filename=f"{chain}_J_gene_distribution.html"))

            mean_chain_vj_df = self._average_norm_counts_per_repertoire(chain_vj_df=vj_df[vj_df["chain"] == chain])


            tables.append(self._write_output_table(mean_chain_vj_df,
                                                   file_path=self.result_path / f"VJ_gene_distribution_averaged_across_repertoires.tsv",
                                                   name=f"Combined V+J gene distribution averaged across repertoires"))

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
                                         filename=f"{chain}_VJ_gene_distribution_averaged_across_repertoires.html",
                                         value_to_plot="mean_norm_counts",
                                         color_name="Average observed frequency<br>across repertoires<br>normalised by repertoire size"))
        else:
            for label_class in set(mean_chain_vj_df[self.label_name]):
                plots.append(self._safe_plot(plot_callable="_plot_gene_combo_heatmap",
                                             chain_df=mean_chain_vj_df[mean_chain_vj_df[self.label_name] == label_class],
                                             title=f"Combined {chain} V+J gene distribution for {self.label_name}={label_class} averaged across repertoires",
                                             filename=f"{chain}_VJ_gene_distribution_{self.label_name}={label_class}_averaged_across_repertoires.html",
                                             zmax=max(mean_chain_vj_df["mean_norm_counts"]),
                                             value_to_plot="mean_norm_counts",
                                             color_name="Average observed frequency<br>across repertoires<br>normalised by repertoire size"))

        return plots

    def _average_norm_counts_per_repertoire(self, chain_vj_df):
        groupby_cols = ["v_genes", "j_genes", "chain"]
        groupby_cols += [self.label_name] if self.label_name is not None else []

        mean_chain_vj_df = pd.DataFrame(chain_vj_df.groupby(groupby_cols)["norm_counts"].mean()).reset_index()
        mean_chain_vj_df.rename(columns={"norm_counts": "mean_norm_counts"}, inplace=True)

        return mean_chain_vj_df

    def _get_repertoire_count_dfs(self):
        v_dfs = []
        j_dfs = []
        vj_dfs = []

        for repertoire in self.dataset.repertoires:
            repertoire_attributes = {"v_call": repertoire.get_v_genes(),
                                     "j_call": repertoire.get_j_genes(),
                                     "chain": repertoire.get_attribute("chain", as_list=True)}

            v_rep_df = self._get_gene_count_df(repertoire_attributes, "v_call", include_label=False)
            j_rep_df = self._get_gene_count_df(repertoire_attributes, "j_call", include_label=False)
            vj_rep_df = self._get_vj_combo_count_df(repertoire_attributes, include_label=False)

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
        rep_df["subject_id"] = repertoire.metadata["subject_id"]
        rep_df["repertoire_size"] = repertoire.get_element_count()
        rep_df["norm_counts"] = rep_df["counts"] / rep_df["repertoire_size"]

        if self.label_name is not None:
            rep_df[self.label_name] = repertoire.metadata[self.label_name]


    def _write_repertoire_tables(self, v_df, j_df, vj_df):
        tables = []

        tables.append(self._write_output_table(v_df,
                                               file_path=self.result_path / f"V_gene_distribution.tsv",
                                               name=f"V gene distribution per repertoire"))

        tables.append(self._write_output_table(j_df,
                                               file_path=self.result_path / f"J_gene_distribution.tsv",
                                               name=f"J gene distribution per repertoire"))

        tables.append(self._write_output_table(vj_df,
                                               file_path=self.result_path / f"VJ_gene_distribution.tsv",
                                               name=f"Combined V+J gene distribution"))

        return tables

    def _plot_gene_distribution_across_repertoires(self, chain_df, title, filename):
        figure = px.box(chain_df, x="genes", y="norm_counts", color=self.label_name,
                        hover_data=["repertoire_id", "subject_id"],
                        color_discrete_sequence=px.colors.diverging.Tealrose)
        figure.update_layout(template="plotly_white")

        file_path = self.result_path / filename
        figure.write_html(str(file_path))
        return ReportOutput(path=file_path, name=title)

    def check_prerequisites(self):
        if self.split_by_label:
            if self.label_name is None:
                if len(self.dataset.get_label_names()) != 1:
                    warnings.warn(f"{VJGeneDistribution.__name__}: ambiguous label: split_by_label was set to True but no label name was specified, and the number of available labels is {len(self.dataset.get_label_names())}: {self.dataset.get_label_names()}. Skipping this report...")
                    return False
            else:
                if self.label_name not in self.dataset.get_label_names():
                    warnings.warn(f"{VJGeneDistribution.__name__}: the specified label name ({self.label_name}) was not available among the dataset labels: {self.dataset.get_label_names()}. Skipping this report...")
                    return False

        if isinstance(self.dataset, ReceptorDataset) or isinstance(self.dataset, SequenceDataset):
            try:
                self.dataset.get_attributes(attributes=["v_call", "j_call", "chain"])
            except AttributeError as e:
                warnings.warn(f"{VJGeneDistribution.__name__}: the following attributes were expected to be present in the dataset: v_call, j_call, chain. The following error occurred when attempting to get these attributes: {e} Skipping this report...")
                return False

        return True

