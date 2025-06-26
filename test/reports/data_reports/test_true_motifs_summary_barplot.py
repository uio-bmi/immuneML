import shutil

from immuneML.IO.dataset_import.AIRRImport import AIRRImport
from immuneML.app.ImmuneMLApp import ImmuneMLApp
from immuneML.data_model.bnp_util import write_yaml
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.TrueMotifsSummaryBarplot import TrueMotifsSummaryBarplot
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_true_motifs_summary_barplot_report():

    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "true_motif_summary")

    sequence_dataset = RandomDatasetGenerator.generate_sequence_dataset(100,
                                                                        {20: 1.},
                                                                        {'model': {'A': 0.5, 'B': 0.5}},
                                                                        path,
                                                                        region_type="IMGT_JUNCTION",)

    gen_model = {"SimpleVAE": {'num_epochs': 10,
                                'latent_dim': 8,
                                'pretrains': 1,
                                'warmup_epochs': 1}}

    generated_model_path = PathBuilder.build(path / str(list(gen_model.keys())[0]))

    specs = {
        "definitions": {
            "datasets": {
                "d1": {
                    "format": "RandomSequenceDataset",
                    "params": {
                        'length_probabilities': {
                            11: 0.5,
                            10: 0.5
                        },
                        'sequence_count': 100,
                        'region_type': 'IMGT_JUNCTION'
                    }
                }
            },
            "ml_methods": {
                'VAE': gen_model
            },
        },
        "instructions": {
            "inst1": {
                "type": "TrainGenModel",
                "gen_examples_count": 100,
                "dataset": "d1",
                "methods": ["VAE"],
                'export_combined_dataset': True
            }
        }
    }

    write_yaml(generated_model_path / 'specs.yaml', specs)
    ImmuneMLApp(generated_model_path / 'specs.yaml', generated_model_path / 'output').run()

    combined_dataset_path = generated_model_path / 'output/inst1/exported_combined_dataset/combined_inst1_dataset.tsv'
    params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "AIRR")
    params["is_repertoire"] = False
    params["paired"] = False
    params["result_path"] = combined_dataset_path.parent
    params["path"] = combined_dataset_path
    params["label_columns"] = ["custom_label"]

    combined_sequence_dataset = AIRRImport(params, "airr_sequence_dataset").import_dataset()

    report = TrueMotifsSummaryBarplot.build_object(**{"dataset": combined_sequence_dataset,
                                              "implanted_motifs_per_signal": {"signal1": {"seeds": ['DE'], "gap_sizes": [0]},
                                                                              "signal2": {"seeds": ['ET'], "gap_sizes": [0]},
                                                                              "signal3": {"seeds": ['S/Y'], "gap_sizes": [1]}},
                                              "result_path": path / "result",
                                              "name": "Test True Motif Summary Report"}
                                                   )

    result = report._generate()

    shutil.rmtree(path)
