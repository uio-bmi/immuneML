
from pathlib import Path
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.GenerativeModel import GenerativeModel
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.ml_reports.UnsupervisedMLReport import UnsupervisedMLReport
from immuneML.util.PathBuilder import PathBuilder
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import logomaker


class GeneratorReport(UnsupervisedMLReport):
    @classmethod
    def build_object(cls, **kwargs):
        name = kwargs["name"] if "name" in kwargs else "GeneratorReport"
        return GeneratorReport(name=name)

    def __init__(self, dataset: Dataset = None, method: GenerativeModel = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, method=method, result_path=result_path,
                         name=name, number_of_processes=number_of_processes)
        self.sequences = None
        self.alphabet = []

    def _make_CharToInt_logo(self, data_logo_path):
        sequences = self.dataset.encoded_data.examples
        sequences_alpha = "".join([self.alphabet[i] for i in sequences]).split(" ")

        sequences_by_length = {}
        for sequence in sequences_alpha:
            if len(sequence) not in sequences_by_length:
                sequences_by_length[len(sequence)] = [sequence]
            else:
                sequences_by_length[len(sequence)].append(sequence)

        new_sequences = []
        for sequences in sequences_by_length.values():
            if len(sequences) > len(new_sequences):
                new_sequences = sequences

        data_counts = logomaker.alignment_to_matrix(sequences=new_sequences, to_type='counts')
        logomaker.Logo(data_counts, color_scheme="dmslogo_funcgroup")
        plt.grid(False)
        plt.savefig(data_logo_path)
    def _make_OneHot_logo(self, data_logo_path):
        sequences_by_length = {}

        for sequence in self.dataset.encoded_data.examples:
            sequence_without_fill = sequence[np.any(sequence == 1, axis=1)]
            if len(sequence_without_fill) not in sequences_by_length:
                sequences_by_length[len(sequence_without_fill)] = [sequence_without_fill]
            else:
                sequences_by_length[len(sequence_without_fill)].append(sequence_without_fill)

        new_sequences = []
        for key in sequences_by_length:
            if len(sequences_by_length[key]) > len(new_sequences):
                new_sequences = sequences_by_length[key]

        sequences = np.array(np.argmax(new_sequences, axis=2))

        alpha_sequences = []
        for sequence in sequences:
            alpha_sequences.append([self.alphabet[i] for i in sequence])

        sequences = []
        for sequence in alpha_sequences:
            sequences.append("".join(sequence))

        data_counts = logomaker.alignment_to_matrix(sequences=sequences, to_type='counts')
        logomaker.Logo(data_counts, color_scheme="dmslogo_funcgroup")
        plt.grid(False)
        plt.savefig(data_logo_path)

    def _make_generated_logo(self, logo_path):

        sequences_dict = {}
        for sequence in self.sequences:
            if len(sequence) not in sequences_dict.keys():
                sequences_dict[len(sequence)] = [sequence]
            else:
                sequences_dict[len(sequence)].append(sequence)

        new_sequences = []
        for key in sequences_dict:
            if len(sequences_dict[key]) > len(new_sequences):
                new_sequences = sequences_dict[key]

        data_counts = logomaker.alignment_to_matrix(sequences=new_sequences, to_type='counts')
        logomaker.Logo(data_counts, color_scheme="dmslogo_funcgroup")
        plt.grid(False)
        plt.savefig(logo_path)

    def _make_report(self) -> ReportResult:
        PathBuilder.build(self.result_path)
        output_figures = []

        generated_sequences = self.result_path / f"{self.name}_GeneratedSequences.tsv"
        logo_path = self.result_path / f"{self.name}_GeneratedData_Logo.png"
        data_logo_path = self.result_path / f"{self.name}_InputData_Logo.png"

        data = pd.DataFrame(enumerate(self.sequences), columns=["id", "sequence_aas"])
        data.to_csv(generated_sequences, index=False, sep='\t')

        self._make_generated_logo(logo_path)
        output_figures.append(ReportOutput(logo_path, name="Generated Logo"))

        if self.dataset:
            if self.dataset.encoded_data.encoding == "OneHotEncoder":
                self._make_OneHot_logo(data_logo_path)
                output_figures.append(ReportOutput(data_logo_path, name="Dataset Logo"))
            elif self.dataset.encoded_data.encoding == "CharToIntEncoder":
                self._make_CharToInt_logo(data_logo_path)
                output_figures.append(ReportOutput(data_logo_path, name="Dataset Logo"))

        sequences_to_output = ReportOutput(generated_sequences, name="Generated Sequences")

        return ReportResult(self.name, output_figures=output_figures, output_tables=[sequences_to_output])

    def _generate(self) -> ReportResult:
        return self._make_report()

