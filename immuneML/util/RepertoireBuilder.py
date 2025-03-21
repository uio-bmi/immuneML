import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import random
import pandas as pd

from immuneML.data_model.AIRRSequenceSet import AIRRSequenceSet
from immuneML.data_model.SequenceSet import ReceptorSequence, Repertoire
from immuneML.data_model.bnp_util import build_dynamic_bnp_dataclass_obj, make_full_airr_seq_set_df, \
    write_dataset_yaml, write_yaml
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.util.PathBuilder import PathBuilder


class RepertoireBuilder:
    """
    Helper class for tests: creates repertoires from a list of a list of sequences and stores them in the given path
    """

    @staticmethod
    def build(sequences: list, path: Path, labels: dict = None, seq_metadata: list = None, subject_ids: list = None,
              name: str = "d1"):

        if subject_ids is not None:
            assert len(subject_ids) == len(sequences)

        if seq_metadata is not None:
            assert len(sequences) == len(seq_metadata)
            for index, sequence_list in enumerate(sequences):
                assert len(sequence_list) == len(seq_metadata[index])

        assert not (
                path / "repertoires").is_dir(), f"RepertoireBuilder: attempted to store new repertoires at {path / 'repertoires'} but this folder already exists. " \
                                                f"Please remove this folder or specify a different path. "

        PathBuilder.build(path)
        rep_path = PathBuilder.build(path / "repertoires")

        repertoires = []
        if subject_ids is None:
            subject_ids = []

        for rep_index, sequence_list in enumerate(sequences):
            if len(subject_ids) < len(sequences):
                subject_ids.append("rep_" + str(rep_index))

            df = pd.DataFrame({
                'cdr3_aa': sequence_list,
                'cdr3': ['' for _ in range(len(sequence_list))],
                'sequence_id': [uuid.uuid4().hex for _ in range(len(sequence_list))],
                'productive': ['T' for _ in range(len(sequence_list))],
                'vj_in_frame': ['T' for _ in range(len(sequence_list))],
                'stop_codon': ['F' for _ in range(len(sequence_list))]
            })

            if seq_metadata is None:
                df['v_call'], df['j_call'], df['duplicate_count'], df['locus'] = "TRBV1-1*01", "TRBJ1-1*01", [1 for _ in sequence_list], "TRB"
            else:
                df = pd.concat([df, pd.DataFrame.from_records(seq_metadata[rep_index])], axis=1)

            df = make_full_airr_seq_set_df(df)

            df.to_csv(str(rep_path / f'rep_{rep_index}.tsv'), sep='\t', index=False)

            if labels is not None:
                metadata = {key: labels[key][rep_index] for key in labels.keys()}
            else:
                metadata = {}

            metadata['type_dict_dynamic_fields'] = {key: AIRRSequenceSet.TYPE_TO_STR[df[key].dtype]
                                                    for key in df.columns
                                                    if key not in AIRRSequenceSet.get_field_type_dict()}

            metadata = {**metadata, **{"subject_id": subject_ids[rep_index]}}
            write_yaml(rep_path / f"rep_{rep_index}.yaml", metadata)

            bnp_dc_obj, _ = build_dynamic_bnp_dataclass_obj(df.to_dict(orient='list'))

            repertoire = Repertoire(rep_path / f"rep_{rep_index}.tsv", rep_path / f"rep_{rep_index}.yaml", metadata,
                                    _bnp_dataclass=type(bnp_dc_obj), identifier=uuid.uuid4().hex)
            repertoires.append(repertoire)

        df = pd.DataFrame({**{"filename": [repertoire.data_filename for repertoire in repertoires],
                              "subject_id": subject_ids,
                              "identifier": [repertoire.identifier for repertoire in repertoires]},
                           **(labels if labels is not None else {})})
        df.to_csv(path / f"{name}_metadata.csv", index=False)

        return repertoires, path / f"{name}_metadata.csv"

    @staticmethod
    def build_dataset(sequences: list, path: Path, labels: dict = None, seq_metadata: list = None,
                      subject_ids: list = None, name: str = "d1"):
        reps, metadata_file = RepertoireBuilder.build(sequences, path, labels, seq_metadata, subject_ids, name)

        # type_dict = {k: v for tmp_dict in [rep.metadata['type_dict_dynamic_fields'] for rep in reps]
        #              for k, v in tmp_dict.items()}

        labels_unique = {k: list(set(v)) for k, v in labels.items()} if isinstance(labels, dict) else {}
        identifier = uuid.uuid4().hex

        metadata_yaml = RepertoireDataset.create_metadata_dict(labels=labels_unique,
                                                               identifier=identifier,
                                                               metadata_file=str(metadata_file.name),
                                                               name=name)

        write_dataset_yaml(path / f'{name}.yaml', metadata_yaml)

        return RepertoireDataset(repertoires=reps, metadata_file=metadata_file, name=name, labels=labels_unique,
                                 dataset_file=path / f'{name}.yaml', identifier=identifier)
