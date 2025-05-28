import os
import shutil

import numpy as np

from immuneML import Constants
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceSet import Repertoire, ReceptorSequence
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.diversity_encoding.ShannonDiversityEncoder import ShannonDiversityEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


def test_encode():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "shannon_div_enc/")

    rep1 = Repertoire.build_from_sequences(
        sequences=[ReceptorSequence(sequence_aa="AAA", duplicate_count=10, vj_in_frame='T') for i in range(1)] +
                  [ReceptorSequence(sequence_aa="AAA", duplicate_count=100, vj_in_frame='T') for i in range(1)] +
                  [ReceptorSequence(sequence_aa="AAA", duplicate_count=1, vj_in_frame='T') for i in range(1)],
        metadata={"l1": "test_1", "l2": 2}, result_path=path)

    rep2 = Repertoire.build_from_sequences(
        sequences=[ReceptorSequence(sequence_aa="AAA", duplicate_count=10) for i in range(3)],
        metadata={"l1": "test_2", "l2": 3}, result_path=path)

    lc = LabelConfiguration()
    lc.add_label("l1", ["test_1", "test_2"])
    lc.add_label("l2", [0, 3])

    dataset = RepertoireDataset.build_from_objects(repertoires=[rep1, rep2], path=path)

    encoder = ShannonDiversityEncoder.build_object(dataset, **{
        'name': 'shannon_encoder'
    })

    d1 = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded/",
        label_config=lc,
    ))

    assert np.allclose(np.round(d1.encoded_data.examples, 3), np.array([0.353, 1.099])), \
        "Encoded examples do not match expected values."

    shutil.rmtree(path)
