from unittest import TestCase

import torch

from immuneML.data_model.SequenceParams import RegionType
from immuneML.encodings.protein_embedding.ProteinEmbeddingEncoder import ProteinEmbeddingEncoder


class StubEmbeddingEncoder(ProteinEmbeddingEncoder):
    """Only the cpu thread setup is under test here, so the model-specific parts are stubbed out."""

    @staticmethod
    def build_object(dataset, **params):
        return StubEmbeddingEncoder(**params)

    def _embed_sequence_set(self, sequence_set, seq_field):
        raise NotImplementedError

    def _get_encoding_name(self) -> str:
        return "stub"

    def _get_model_link(self) -> str:
        return "stub"


class TestProteinEmbeddingEncoder(TestCase):

    def setUp(self) -> None:
        self.original_num_threads = torch.get_num_threads()

    def test_num_threads_is_applied_on_cpu(self):
        encoder = StubEmbeddingEncoder(region_type=RegionType.IMGT_CDR3, name='stub', num_threads=2, device='cpu')
        encoder._set_up_cpu_threads()

        self.assertEqual(torch.get_num_threads(), 2)

    def test_default_leaves_torch_setting_alone(self):
        # with the default of 1, torch keeps using all available cores: capping it to a single thread would
        # slow down every analysis that does not set the parameter
        encoder = StubEmbeddingEncoder(region_type=RegionType.IMGT_CDR3, name='stub', num_threads=1, device='cpu')
        encoder._set_up_cpu_threads()

        self.assertEqual(torch.get_num_threads(), self.original_num_threads)

    def test_other_devices_are_not_affected(self):
        encoder = StubEmbeddingEncoder(region_type=RegionType.IMGT_CDR3, name='stub', num_threads=2, device='cuda')
        encoder._set_up_cpu_threads()

        self.assertEqual(torch.get_num_threads(), self.original_num_threads)

    def tearDown(self) -> None:
        torch.set_num_threads(self.original_num_threads)
