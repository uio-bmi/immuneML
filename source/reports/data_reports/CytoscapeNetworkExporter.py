import warnings
import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.reports.data_reports.DataReport import DataReport
from source.util.ParameterValidator import ParameterValidator
from source.util.PathBuilder import PathBuilder


class CytoscapeNetworkExporter(DataReport):
    """
    This report exports the Receptor sequences to .sif format, such that they can directly be
    imported as a network in Cytoscape, to visualize chain sharing between the different receptors
    in a dataset.
    The Receptor sequences can be provided as a ReceptorDataset, or a RepertoireDataset (containing paired sequence
    information). In the latter case, one .sif file is exported per Repertoire.

    Specification:

        definitions:
            datasets:
                my_receptor_dataset:
                    format: ...
                    params:
                        path: path/to/receptors/
                        result_path: path/for/results/

                my_repertoire_dataset:
                    format: ...
                    params:
                        path: path/to/repertoires/
                        result_path: path/for/results/
                        metadata_file: path/to/metadata.txt
                        paired: True

            reports:
                my_cyto_export: CytoscapeNetworkExporter

        instructions:
                instruction_1:
                    type: ExploratoryAnalysis
                    analyses:
                        my_analysis_1:
                            dataset: my_receptor_dataset
                            report: my_cyto_export
                        my_analysis_2:
                            dataset: my_repertoire_dataset
                            report: my_cyto_export
    """

    CHAIN_GENE_NAME_CONVERSION = {"A": "TRA",
                                  "B": "TRB",
                                  "G": "TRG",
                                  "D": "TRD",
                                  "H": "IGH",
                                  "L": "IGL"}


    @classmethod
    def build_object(cls, **kwargs):
        print(kwargs["additional_attributes"])

        if kwargs["additional_attributes"] is None:
            kwargs["additional_attributes"] = []


        ParameterValidator.assert_type_and_value(kwargs["additional_attributes"], list, "CytoscapeNetworkExporter", "additional_attributes")

        return CytoscapeNetworkExporter(**kwargs)

    def __init__(self, dataset: Dataset = None, result_path: str = None,
                 chains=("alpha", "beta"), drop_duplicates = True, additional_attributes=[]):
        self.chains = chains
        self.drop_duplicates = drop_duplicates
        self.additional_attributes = additional_attributes
        DataReport.__init__(self, dataset=dataset, result_path=result_path)


    def check_prerequisites(self):
        if isinstance(self.dataset, RepertoireDataset) or isinstance(self.dataset, ReceptorDataset):
            return True
        else:
            warnings.warn("CytoscapeNetworkExporter: report can be generated only from a ReceptorDataset or a RepertoireDataset (with repertoires containing Receptors). Skipping this report...")
            return False

    def generate(self):
        if isinstance(self.dataset, RepertoireDataset):
            for repertoire in self.dataset.get_data():
                result_path = f"{self.result_path}/{repertoire.identifier}/"
                PathBuilder.build(result_path)
                self.export_receptorlist(repertoire.receptors, result_path)
        elif isinstance(self.dataset, ReceptorDataset):
            receptors = self.dataset.get_data()
            result_path = f"{self.result_path}/{self.dataset.identifier}/"
            PathBuilder.build(result_path)
            self.export_receptorlist(receptors, result_path=result_path)


    def export_receptorlist(self, receptors, result_path):
        export_list = []
        metadata_list = []

        for receptor in receptors:
            first_chain = receptor.get_chain(self.chains[0])
            second_chain = receptor.get_chain(self.chains[1])
            first_chain_name = self.get_shared_name(first_chain)
            second_chain_name = self.get_shared_name(second_chain)

            export_list.append([first_chain_name, "(pair)", second_chain_name])

            metadata_list.append([first_chain_name, self.chains[0]] + self.get_formatted_metadata(first_chain))
            metadata_list.append([second_chain_name, self.chains[1]] + self.get_formatted_metadata(second_chain))

        full_df = pd.DataFrame(export_list, columns=[self.chains[0], "relationship", self.chains[1]])
        meta_df = pd.DataFrame(metadata_list, columns=["shared_name", "chain", "sequence",
                                                       "v_subgroup", "v_gene",
                                                       "j_subgroup", "j_gene"] + self.additional_attributes)

        meta_df.drop_duplicates(inplace=True)
        meta_df.to_csv(f"{result_path}metadata.tsv", sep="\t", index=0, header=True)

        if self.drop_duplicates:
            full_df.drop_duplicates(inplace=True)

        full_df.to_csv(f"{result_path}all_chains.sif", sep="\t", index=0, header=False)

        shared_df = full_df[(full_df.duplicated(["alpha"], keep=False)) | (full_df.duplicated(["beta"], keep=False))]
        shared_df.to_csv(f"{result_path}shared_chains.sif", sep="\t", index=0, header=False)


    def get_shared_name(self, seq: ReceptorSequence):
        '''Returns a string containing a representation of the given receptor chain, with
        the chain, sequence, v and j genes.
        For example: *a*s=AMREGPEHSGYALN*v=V7-3*j=J41'''
        return f"*{seq.get_attribute('chain').value.lower()}" \
               f"*s={seq.get_sequence()}" \
               f"*v={seq.get_attribute('v_gene')}" \
               f"*j={seq.get_attribute('j_gene')}"

    def get_formatted_metadata(self, seq: ReceptorSequence):
        # sequence, v_gene_subgroup, v_gene, j_gene_subgroup, j_gene
        chain = CytoscapeNetworkExporter.CHAIN_GENE_NAME_CONVERSION[seq.get_attribute('chain').value]
        v_gene = seq.get_attribute('v_gene')
        j_gene = seq.get_attribute('j_gene')

        additional_info = []

        for attr in self.additional_attributes:
            try:
                additional_info.append(seq.get_attribute(attr))
            except KeyError:
                additional_info.append(None)
                warnings.warn(f"CytoscapeNetworkExporter: additional metadata attribute {attr} was not found for some receptor chain(s), value None was used instead.")

        return [seq.get_sequence(),
               f"{chain}{v_gene.split('-')[0]}", f"{chain}{v_gene}",
               f"{chain}{j_gene.split('-')[0]}", f"{chain}{j_gene}"] + additional_info
