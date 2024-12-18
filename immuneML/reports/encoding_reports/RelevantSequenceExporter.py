import logging
import os
from pathlib import Path

import pandas as pd

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceParams import RegionType
from immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder import CompAIRRSequenceAbundanceEncoder
from immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.PathBuilder import PathBuilder


class RelevantSequenceExporter(EncodingReport):
    """
    Exports the sequences that are extracted as label-associated when using the :py:obj:`~immuneML.encodings.abundance_encoding.SequenceAbundanceEncoder.SequenceAbundanceEncoder` or
    :py:obj:`~immuneML.encodings.abundance_encoding.CompAIRRSequenceAbundanceEncoder.CompAIRRSequenceAbundanceEncoder` in AIRR-compliant format.


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            reports:
                my_relevant_sequences: RelevantSequenceExporter

    """

    def __init__(self, dataset: RepertoireDataset = None, result_path: Path = None, name: str = None, number_of_processes: int = 1):
        super().__init__(dataset=dataset, result_path=result_path, name=name, number_of_processes=number_of_processes)

    @classmethod
    def build_object(cls, **kwargs):
        return RelevantSequenceExporter(**kwargs)

    def _generate(self) -> ReportResult:

        df = pd.read_csv(self.dataset.encoded_data.info["relevant_sequence_path"])

        PathBuilder.build(self.result_path)
        filename = self.result_path / "relevant_sequences.csv"
        df.to_csv(filename, index=False)

        return ReportResult(self.name,
                            info=f"Exports the sequences that are extracted as label-associated using the {self.dataset.encoded_data.encoding} in AIRR-compliant format.",
                            output_tables=[ReportOutput(filename, "relevant sequences")])

    def check_prerequisites(self):
        valid_encodings = [SequenceAbundanceEncoder.__name__, CompAIRRSequenceAbundanceEncoder.__name__]
        if self.dataset.encoded_data is None or self.dataset.encoded_data.info is None:
            logging.warning("RelevantSequenceExporter: the dataset is not encoded, skipping this report...")
            return False
        elif self.dataset.encoded_data.encoding not in valid_encodings:
            logging.warning(f"RelevantSequenceExporter: the dataset encoding ({self.dataset.encoded_data.encoding}) was not in the list of valid "
                            f"encodings ({valid_encodings}), skipping this report...")
            return False
        elif "relevant_sequence_path" not in self.dataset.encoded_data.info or not os.path.isfile(self.dataset.encoded_data.info['relevant_sequence_path']):
            logging.warning(f"RelevantSequenceExporter: the relevant sequences were not set for this encoded data, skipping this report...")
            return False
        else:
            return True
