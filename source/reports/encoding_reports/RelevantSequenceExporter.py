import logging
import os

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.filtered_sequence_encoding.SequenceAbundanceEncoder import SequenceAbundanceEncoder
from source.encodings.filtered_sequence_encoding.SequenceCountEncoder import SequenceCountEncoder
from source.reports.ReportOutput import ReportOutput
from source.reports.ReportResult import ReportResult
from source.reports.encoding_reports.EncodingReport import EncodingReport
from source.util.PathBuilder import PathBuilder


class RelevantSequenceExporter(EncodingReport):
    """
    Exports the sequences that are extracted as label-associated using `SequenceAbundance` or `SequenceCount` encoders in AIRR-compliant format.

    Arguments: there are no arguments for this report.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_relevant_sequences: RelevantSequenceExporter

    """

    COLUMN_MAPPING = {
        "v_genes": "v_call",
        "j_genes": "j_call",
        "sequences": "cdr3",
        'sequence_aas': "cdr3_aa"
    }

    def __init__(self, dataset: RepertoireDataset = None, result_path: str = None, name: str = None):
        super().__init__(name)
        self.dataset = dataset
        self.result_path = result_path

    @classmethod
    def build_object(cls, **kwargs):
        return RelevantSequenceExporter(**kwargs)

    def generate(self) -> ReportResult:

        df = pd.read_csv(self.dataset.encoded_data.info["relevant_sequence_path"])
        column_mapping = self._compute_column_mapping(df)
        df.rename(columns=column_mapping, inplace=True)

        PathBuilder.build(self.result_path)
        filename = f"{self.result_path}relevant_sequences.csv"
        df.to_csv(filename, index=False)

        return ReportResult(self.name, output_tables=[ReportOutput(filename, "relevant sequences")])

    def _compute_column_mapping(self, df: pd.DataFrame) -> dict:
        columns = df.columns.values.tolist()
        column_mapping = {}
        region_type = self.dataset.get_repertoire(0).get_attribute("region_types")[0]
        if "sequence_aas" in columns and region_type != "CDR3":
            column_mapping["sequence_aas"] = "sequence_aa"
        if "sequences" in columns and region_type != "CDR3":
            column_mapping['sequences'] = "sequence"

        return {**RelevantSequenceExporter.COLUMN_MAPPING, **column_mapping}

    def check_prerequisites(self):
        valid_encodings = [SequenceAbundanceEncoder.__name__, SequenceCountEncoder.__name__]
        if self.dataset.encoded_data is None or self.dataset.encoded_data.examples is None:
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
