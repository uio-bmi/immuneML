import random
import shutil
import string
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.encoding_reports.heatmap.SimilarityHeatmap import SimilarityHeatmap


class TestSimilarityHeatmap(TestCase):

    def get_group(self, index):
        if index < 20:
            return "A"
        elif 20 <= index < 40:
            return "B"
        elif 40 <= index < 60:
            return "C"
        elif 60 <= index < 80:
            return "D"
        else:
            return "E"

    def test_generate(self):

        path = EnvironmentSettings.root_path + "test/tmp/similarityheatmap/"

        sequences = [''.join(random.choices(string.ascii_uppercase, k=12)) for i in range(30)]

        encoded_data = {
            'examples': sparse.csr_matrix(np.random.normal(50, 10, 15000).reshape((500, 30))),
            'example_ids': [''.join(random.choices(string.ascii_uppercase, k=4)) for i in range(500)],
            'labels': {
                "patient": np.array([i for i in range(100) for _ in range(5)]),
                "week": np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100),
                "disease": np.array((["T1D"] * 125 + ["CONTROL"] * 125 + ["SLE"] * 125 + ["RA"] * 125)),
                "age": np.random.normal(50, 10, 500),
            },
            'feature_names': sequences,
            'feature_annotations': pd.DataFrame({
                "sequence": sequences,
                "antigen": ["GAD"] * 15 + ["INSB"] * 15
            })
        }

        dataset = RepertoireDataset(encoded_data=EncodedData(**encoded_data))

        SimilarityHeatmap.build_object(dataset=dataset,
                                       similarity_measure="PEARSON",
                                       one_hot_encode_example_annotations=["disease"],
                                       example_annotations=["age", "week"],
                                       palette={"week": {"4": ["white"]},
                                                "age": {"colors": ["cornflowerblue", "white", "firebrick"], "breaks": [40, 50, 60]}},
                                       annotation_position="left",
                                       show_names=True, names_size=0.5, height=5, width=6.7, text_size=6,
                                       result_path=path).generate_report()

        shutil.rmtree(path)
