import os
import shutil
import sys
from pathlib import Path

import yaml

from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class Quickstart:

    def create_specfication(self, path: Path):

        specs = {
            "definitions": {
                "datasets": {
                    "d1": {
                        "format": "AIRR",
                        "params": {
                            "path": str(path / "../synthetic_dataset/result/my_simulation_instruction/exported_dataset/airr/"),
                            "metadata_file": str(path / "../synthetic_dataset/result/my_simulation_instruction/exported_dataset/airr/metadata.csv")
                        }
                    }
                },
                "encodings": {
                    "e1": {
                        "KmerFrequency": {
                            "k": 3
                        }
                    }
                },
                "reports": {
                    "rep1": {
                        "FeatureDistribution": {
                            "mode": "sparse",
                        }
                    }
                }
            },
            "instructions": {
                "inst1": {
                    "type": "ExploratoryAnalysis",
                    "analyses": {
                        "analysis_1": {
                            "dataset": "d1",
                            "encoding": "e1",
                            "report": "rep1"
                        }
                    }
                }
            }
        }
        PathBuilder.build(path)
        specs_file = path / "specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(specs, file)

        return specs_file

    def build_path(self, path: str = None):
        if path is None:
            path = EnvironmentSettings.root_path / "quickstart/"
            if os.path.isdir(path):
                shutil.rmtree(path)
            PathBuilder.build(path)
        else:
            path = Path(path)
        return path

    def _simulate_dataset_with_signals(self, path: Path):

        print("immuneML quickstart: generating a synthetic dataset...")

        PathBuilder.build(path)

        specs = {
            "definitions": {
                "datasets": {
                    "my_synthetic_dataset": {"format": "RandomRepertoireDataset", "params": {"labels": {}}}
                },
                "motifs": {"my_motif": {"seed": "AA", "instantiation": "GappedKmer"}},
                "signals": {"my_signal": {"motifs": ["my_motif"], "implanting": "HealthySequence"}},
                "simulations": {"my_simulation": {"my_implantng": {"signals": ["my_signal"], "dataset_implanting_rate": 0.5,
                                                                   "repertoire_implanting_rate": 0.1}}}
            },
            "instructions": {"my_simulation_instruction": {"type": "Simulation", "dataset": "my_synthetic_dataset", "simulation": "my_simulation",
                                                           "export_formats": ["AIRR"]}}
        }

        specs_file = path / "simulation_specs.yaml"
        with specs_file.open("w") as file:
            yaml.dump(specs, file)

        app = ImmuneMLApp(specs_file, path / "result")
        app.run()

        print("immuneML quickstart: finished generating a synthetic dataset.")

    def run(self, result_path: str):

        result_path = self.build_path(result_path)

        self._simulate_dataset_with_signals(result_path / "synthetic_dataset")

        print("immuneML quickstart: training a machine learning model...")
        specs_file = self.create_specfication(result_path / "quickstart")
        app = ImmuneMLApp(specs_file, result_path / "quickstart/result")
        app.run()

        print("immuneML quickstart: finished training a machine learning model.")


def main():
    quickstart = Quickstart()
    quickstart.run(sys.argv[1] if len(sys.argv) == 2 else None)


if __name__ == "__main__":
    main()
