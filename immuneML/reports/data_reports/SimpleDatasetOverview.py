from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder


class SimpleDatasetOverview(DataReport):
    """
    Generates a simple text-based overview of the properties of any dataset, including the dataset name, size, and metadata labels.

    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_overview: SimpleDatasetOverview

    """
    UNKNOWN_CHAIN = "unknown"

    def __init__(self, dataset: Dataset = None, result_path: Path = None, number_of_processes: int = 1, name: str = None):
        super().__init__(dataset=dataset, result_path=result_path, number_of_processes=number_of_processes, name=name)

    @classmethod
    def build_object(cls, **kwargs):
        return SimpleDatasetOverview(**kwargs)

    def _generate(self) -> ReportResult:
        PathBuilder.build(self.result_path)

        text_path = self.result_path / "dataset_description.txt"

        dataset_name = self.dataset.name if self.dataset.name is not None else self.dataset.identifier

        output_text = self._get_generic_dataset_text()

        if isinstance(self.dataset, RepertoireDataset):
            output_text += self._get_repertoire_dataset_text()
        elif isinstance(self.dataset, ReceptorDataset):
            output_text += self._get_receptor_dataset_text()
        elif isinstance(self.dataset, SequenceDataset):
            output_text += self._get_sequence_dataset_text()

        text_path.write_text(output_text)

        return ReportResult(name=self.name,
                            info=f"A simple overview of the properties of dataset {self.dataset.name}",
                            output_text=[ReportOutput(text_path, f"Description of dataset {dataset_name}")])

    def _get_generic_dataset_text(self):
        element_name = type(self.dataset).__name__.replace("Dataset", "s").lower()

        output_text = f"Dataset name: {self.dataset.name}\n" \
                      f"Dataset identifier: {self.dataset.identifier}\n" \
                      f"Dataset type: {type(self.dataset).__name__}\n" \
                      f"Dataset size: {self.dataset.get_example_count()} {element_name}\n" \
                      f"Labels available for classification:"

        if len(self.dataset.get_label_names()) == 0:
            output_text += " None"
        else:
            for label in self.dataset.get_label_names():
                output_text += "\n - " + label

        return output_text

    def _get_repertoire_dataset_text(self):
        output_text = f"\nmetadata file location: {self.dataset.metadata_file}\n"

        output_text += "\n\nProperties per repertoire:\n"
        for repertoire in self.dataset.repertoires:
            output_text += f"- Name: {repertoire.data_filename.name}\n"
            output_text += f"  Number of sequences: {repertoire.get_element_count()}\n"

            chains = [chain.value if chain else SimpleDatasetOverview.UNKNOWN_CHAIN for chain in set(repertoire.get_chains())]
            if len(chains) == 1:
                output_text += f"  Chain type: {chains[0]}\n"
            else:
                output_text += f"  Chain types: {','.join(chains)}\n"

        return output_text

    def _get_receptor_dataset_text(self):
        receptor_types = list(set([type(receptor).__name__ for receptor in self.dataset.get_data()]))

        if len(receptor_types) > 1:
            output_text = "\nReceptor types: " + ",".join(receptor_types)
        else:
            output_text = "\nReceptor type: " + receptor_types[0]

        return output_text

    def _get_sequence_dataset_text(self):
        chains = list(set([sequence.get_attribute("chain") for sequence in self.dataset.get_data()]))
        chains = [chain.value if chain else SimpleDatasetOverview.UNKNOWN_CHAIN for chain in chains]

        if len(chains) > 1:
            output_text = "\nChain types: " + ",".join(chains)
        else:
            output_text = "\nChain type: " + chains[0]

        return output_text