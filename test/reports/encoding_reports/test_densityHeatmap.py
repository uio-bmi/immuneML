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
from source.reports.encoding_reports.heatmap.DensityHeatmap import DensityHeatmap


class TestDensityHeatmap(TestCase):

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

        path = EnvironmentSettings.root_path + "test/tmp/densityheatmap/"

        sequences = [''.join(random.choices(string.ascii_uppercase, k=12)) for i in range(30)]

        encoded_data = {
            'examples': sparse.csr_matrix(np.random.normal(50, 10, 500 * 30).reshape((500, 30))),
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

        DensityHeatmap.build_object(dataset=dataset,
                                    scale_features=False,
                                    feature_annotations=["antigen"],
                                    palette={"week": {"0": "#BE9764"}, "antigen": {"GAD": "cornflowerblue", "INSB": "firebrick"},
                                             "age": {"colors": ["blue", "white", "red"], "breaks": [0, 20, 100]}},
                                    result_path=path,
                                    show_feature_names=True,
                                    feature_names_size=7,
                                    text_size=9,
                                    height=6,
                                    width=6).generate_report()

        shutil.rmtree(path)
