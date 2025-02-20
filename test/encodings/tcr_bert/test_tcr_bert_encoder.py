import numpy as np

from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.tcr_bert.TCRBertEncoder import TCRBertEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestTCRBertEncoder:

    def _prepare_test_dataset(self, path):
        sequences = [ReceptorSequence(sequence_aa="AAC", sequence="AAACCC", sequence_id="1",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="ACA", sequence="ACACAC", sequence_id="2",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="TCAA", sequence="CCCAAA", sequence_id="3",
                                      metadata={"l1": 1})]

        PathBuilder.build(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = SequenceDataset.build_from_objects(sequences, path)
        return dataset, lc

    def test(self):
        path = EnvironmentSettings.tmp_test_path / "tcr_bert/"
        PathBuilder.remove_old_and_build(path)

        dataset, lc = self._prepare_test_dataset(path)

        encoder = TCRBertEncoder.build_object(dataset=dataset, **{"model": "tcr-bert", 'layers': [-1],
                                                                  "method": "mean", "batch_size": 1})

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True

        ))
        assert isinstance(encoded_dataset.encoded_data.examples, np.ndarray), "The embeddings are not of type numpy.ndarray"
        assert encoded_dataset.encoded_data.examples.shape == (3, 768), f"The array shape is {encoded_dataset.encoded_data.examples.shape}, expected (3, 768)"
