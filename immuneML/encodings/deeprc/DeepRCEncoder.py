from pathlib import Path

import pandas as pd

from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.PathBuilder import PathBuilder


class DeepRCEncoder(DatasetEncoder):
    """
    DeepRCEncoder should be used in combination with the DeepRC ML method (:ref:`DeepRC`).
    This encoder writes the data in a RepertoireDataset to .tsv files.
    For each repertoire, one .tsv file is created containing the amino acid sequences and the counts.
    Additionally, one metadata .tsv file is created, which describes the subset of repertoires that is encoded by
    a given instance of the DeepRCEncoder.

    Note: sequences where count is None, the count value will be set to 1

    **Dataset type:**

    - RepertoireDatasets


    **YAML specification:**

    .. indent with spaces
    .. code-block:: yaml

        definitions:
            encodings:
                my_deeprc_encoder: DeepRC

    """
    ID_COLUMN = "ID"
    SEQUENCE_COLUMN = "amino_acid"
    COUNTS_COLUMN = "templates"
    SEP = "\t"
    EXTENSION = "tsv"
    METADATA_EXTENSION = "csv"
    METADATA_SEP = ","

    def __init__(self, context: dict = None, name: str = None):
        super().__init__(name=name)
        self.context = context
        self.max_sequence_length = 0

    def set_context(self, context: dict):
        self.context = context
        return self

    @staticmethod
    def _prepare_parameters(context: dict = None, name: str = None):
        return {"context": context,
                "name": name}

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            prepared_params = DeepRCEncoder._prepare_parameters(**params)
            return DeepRCEncoder(**prepared_params)
        else:
            raise ValueError("DeepRCEncoder is not defined for dataset types which are not RepertoireDataset.")

    def export_repertoire_tsv_files(self, output_folder: Path, params: EncoderParams):
        repertoires = self.context["dataset"].repertoires
        sequence_column = params.region_type.value + ("_aa" if params.sequence_type == SequenceType.AMINO_ACID else "")

        for repertoire in repertoires:
            filepath = output_folder / f"{repertoire.identifier}.{DeepRCEncoder.EXTENSION}"

            if not filepath.is_file():
                df = (repertoire.data.topandas()[[sequence_column, 'duplicate_count']]
                      .rename(columns={sequence_column: DeepRCEncoder.SEQUENCE_COLUMN,
                                       'duplicate_count': DeepRCEncoder.COUNTS_COLUMN}))
                df[DeepRCEncoder.COUNTS_COLUMN].fillna(1, inplace=True)

                df.to_csv(path_or_buf=filepath, sep=DeepRCEncoder.SEP, index=False)

                max_sequence_length = max(df[DeepRCEncoder.SEQUENCE_COLUMN].str.len())
                self.max_sequence_length = max(self.max_sequence_length, max_sequence_length)

    def export_metadata_file(self, dataset, labels, output_folder):
        metadata_filepath = output_folder / f"{dataset.identifier}_metadata.{DeepRCEncoder.METADATA_EXTENSION}"
        metadata = dataset.get_metadata(labels, return_df=True)
        metadata[DeepRCEncoder.ID_COLUMN] = dataset.get_repertoire_ids()

        metadata.to_csv(path_or_buf=metadata_filepath, sep=DeepRCEncoder.METADATA_SEP, index=False)

        return metadata_filepath

    def encode(self, dataset, params: EncoderParams) -> RepertoireDataset:
        result_path = params.result_path / "encoding"
        PathBuilder.build(result_path)

        self.export_repertoire_tsv_files(result_path, params)

        metadata_filepath = self.export_metadata_file(dataset, params.label_config.get_labels_by_name(), result_path)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(examples=None,
                                                   labels=EncoderHelper.encode_dataset_labels(dataset,
                                                                                              params.label_config,
                                                                                              params.encode_labels),
                                                   example_ids=dataset.get_repertoire_ids(),
                                                   encoding=DeepRCEncoder.__name__,
                                                   info={"metadata_filepath": metadata_filepath,
                                                         "max_sequence_length": self.max_sequence_length})

        return encoded_dataset
