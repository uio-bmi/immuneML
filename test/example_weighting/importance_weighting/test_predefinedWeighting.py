import os
import shutil
from unittest import TestCase
import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.predefined_weighting.PredefinedWeighting import PredefinedWeighting
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestPredefinedWeighting(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path: str):
        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                                        {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, labels={"l1": [0, 1], "l2": [2, 3]},
                                    metadata_file=metadata)

        weights_path = path / "mock_weights.tsv"

        df = pd.DataFrame({"identifier": dataset.get_example_ids() + ["missing1", "missing2"],
                           "example_weight": [i for i in range(8)]})
        df.to_csv(weights_path, index=False)

        return dataset, weights_path

    def test_compute_weights(self):
        path = EnvironmentSettings.tmp_test_path / "positional_motif_sequence_encoder/test/"
        dataset, weights_path = self._prepare_dataset(path)

        importance_weighter = PredefinedWeighting.build_object(dataset,
                                                               **{"separator": ",",
                                                                  "file_path": weights_path}
                                                               )

        w = importance_weighter.compute_weights(dataset, ExampleWeightingParams(result_path=path))

        self.assertEqual(importance_weighter.file_path, weights_path)
        self.assertEqual(w, [i for i in range(6)])

        shutil.rmtree(path)

