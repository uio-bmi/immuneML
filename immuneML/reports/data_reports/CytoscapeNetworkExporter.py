import warnings
from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder


class CytoscapeNetworkExporter(DataReport):
    """
    This report exports the Receptor sequences to .sif format, such that they can directly be
    imported as a network in Cytoscape, to visualize chain sharing between the different receptors
    in a dataset (for example, for TCRs: how often one alpha chain is shared with multiple beta chains, and vice versa).

    The Receptor sequences can be provided as a ReceptorDataset, or a RepertoireDataset (containing paired sequence
    information). In the latter case, one .sif file is exported per Repertoire.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_cyto_export: CytoscapeNetworkExporter

    """

    @classmethod
    def build_object(cls, **kwargs):

        if kwargs["additional_node_attributes"] is None:
            kwargs["additional_node_attributes"] = []
        if kwargs["additional_edge_attributes"] is None:
            kwargs["additional_edge_attributes"] = []

        ParameterValidator.assert_type_and_value(kwargs["additional_node_attributes"], list, "CytoscapeNetworkExporter", "additional_node_attributes")
        ParameterValidator.assert_type_and_value(kwargs["additional_edge_attributes"], list, "CytoscapeNetworkExporter", "additional_edge_attributes")

        return CytoscapeNetworkExporter(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: Path = None,
                 chains=("alpha", "beta"), drop_duplicates=True,
                 additional_node_attributes=[], additional_edge_attributes=[], name: str = None,):
        super().__init__(dataset=dataset, result_path=result_path, name=name)
        self.chains = chains
        self.drop_duplicates = drop_duplicates
        self.additional_node_attributes = additional_node_attributes
        self.additional_edge_attributes = additional_edge_attributes

    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset) or isinstance(self.dataset, ReceptorDataset):
            return True
        else:
            warnings.warn(
                "CytoscapeNetworkExporter: report can be generated only from a ReceptorDataset or a RepertoireDataset "
                "(with repertoires containing Receptors). Skipping this report...")
            return False

    def _generate(self):

        report_output_tables = []

        if isinstance(self.dataset, RepertoireDataset):
            for repertoire in self.dataset.get_data():
                result_path = self.result_path / repertoire.identifier
                PathBuilder.build(result_path)
                report_output_tables = self.export_receptorlist(repertoire.receptors, result_path)
        elif isinstance(self.dataset, ReceptorDataset):
            receptors = self.dataset.get_data()
            result_path = self.result_path / self.dataset.identifier
            PathBuilder.build(result_path)
            report_output_tables = self.export_receptorlist(receptors, result_path=result_path)

        return ReportResult(output_tables=report_output_tables)

    def export_receptorlist(self, receptors, result_path: Path):
        export_list = []
        node_metadata_list = []
        edge_metadata_list = []

        for receptor in receptors:
            first_chain = receptor.get_chain(self.chains[0])
            second_chain = receptor.get_chain(self.chains[1])
            first_chain_name = self.get_shared_name(first_chain)
            second_chain_name = self.get_shared_name(second_chain)

            export_list.append([first_chain_name, "pair", second_chain_name])

            node_metadata_list.append([first_chain_name, self.chains[0]] + self.get_formatted_node_metadata(first_chain))
            node_metadata_list.append([second_chain_name, self.chains[1]] + self.get_formatted_node_metadata(second_chain))

            edge_metadata_list.append(
                [f"{first_chain_name} (pair) {second_chain_name}"] + self.get_formatted_edge_metadata(first_chain, second_chain))

        full_df = pd.DataFrame(export_list, columns=[self.chains[0], "relationship", self.chains[1]])
        node_meta_df = pd.DataFrame(node_metadata_list, columns=["shared_name", "chain", "sequence", "v_subgroup", "v_gene", "j_subgroup",
                                                                 "j_gene"] + self.additional_node_attributes)
        edge_meta_df = pd.DataFrame(edge_metadata_list, columns=["shared_name"] + self.additional_edge_attributes)

        node_cols = list(node_meta_df.columns)
        node_meta_df["n_duplicates"] = 1
        node_meta_df = node_meta_df.groupby(node_cols, as_index=False)["n_duplicates"].sum()

        edge_meta_df.drop_duplicates(inplace=True)
        node_meta_df.to_csv(result_path / "node_metadata.tsv", sep="\t", index=0, header=True)
        edge_meta_df.to_csv(result_path / "edge_metadata.tsv", sep="\t", index=0, header=True)

        if self.drop_duplicates:
            full_df.drop_duplicates(inplace=True)

        full_df.to_csv(result_path / "all_chains.sif", sep="\t", index=0, header=False)

        shared_df = full_df[(full_df.duplicated(["alpha"], keep=False)) | (full_df.duplicated(["beta"], keep=False))]
        shared_df.to_csv(result_path / "shared_chains.sif", sep="\t", index=0, header=False)

        return [ReportOutput(path=result_path / "node_metadata.tsv"),
                ReportOutput(path=result_path / "edge_metadata.tsv"),
                ReportOutput(path=result_path / "all_chains.sif"),
                ReportOutput(path=result_path / "shared_chains.sif")]

    def get_shared_name(self, seq: ReceptorSequence):
        """Returns a string containing a representation of the given receptor chain, with the chain, sequence, v and j genes.
        For example: *a*s=AMREGPEHSGYALN*v=V7-3*j=J41"""
        return f"*{seq.get_attribute('chain').value.lower()}" \
               f"*s={seq.get_sequence()}" \
               f"*v={seq.get_attribute('v_gene')}" \
               f"*j={seq.get_attribute('j_gene')}"

    def get_formatted_node_metadata(self, seq: ReceptorSequence):
        # sequence, v_gene_subgroup, v_gene, j_gene_subgroup, j_gene
        chain = seq.get_attribute('chain').value
        v_gene = seq.get_attribute('v_gene')
        j_gene = seq.get_attribute('j_gene')

        additional_info = []

        for attr in self.additional_node_attributes:
            try:
                additional_info.append(seq.get_attribute(attr))
            except KeyError:
                additional_info.append(None)
                warnings.warn(
                    f"CytoscapeNetworkExporter: additional metadata attribute {attr} was not found for some receptor chain(s), "
                    f"value None was used instead.")

        return [seq.get_sequence(),
                f"{chain}{v_gene.split('-')[0]}", f"{chain}{v_gene}",
                f"{chain}{j_gene.split('-')[0]}", f"{chain}{j_gene}"] + additional_info

    def get_formatted_edge_metadata(self, seq1, seq2):
        additional_info = []

        for attr in self.additional_edge_attributes:
            try:
                info1, info2 = seq1.get_attribute(attr), seq2.get_attribute(attr)
                if info1 == info2:
                    additional_info.append(info1)
                else:
                    additional_info.append("|".join([info1, info2]))
            except KeyError:
                additional_info.append(None)
                warnings.warn(
                    f"CytoscapeNetworkExporter: additional metadata attribute {attr} was not found for some receptor, "
                    f"value None was used instead.")

        return additional_info
