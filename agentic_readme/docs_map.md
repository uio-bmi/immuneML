# Documentation map (`docs/source/`, Galaxy excluded)

**How the docs are generated**
- `docs/specs/` is generated from class docstrings by `scripts/specs_docs_generation.py`. Run it from the repo root with `PYTHONPATH=.`.
- CI regenerates and publishes the docs on every push to `development`. A local `docs/specs` folder can be stale.
- The pages under `yaml_specs/*.rst` only `include` those generated files.

## Pages to start from

| Need | Page |
|---|---|
| Spec structure overview | `yaml_specs/how_to_specify_an_analysis_with_yaml.rst` |
| Training a classifier, full example | `tutorials/how_to_train_and_assess_a_receptor_or_repertoire_classifier.rst` |
| End-to-end analysis of your own data (exploration, subsampling, ML) | `tutorials/analyze_dataset.rst` |
| Importing data | `tutorials/how_to_import_the_data_to_immuneML.rst` |
| Exploratory analysis | `tutorials/how_to_perform_exploratory_analysis.rst` |
| Applying a trained model | `tutorials/how_to_apply_to_new_data.rst` |
| Combining encodings | `tutorials/how_to_combine_multiple_encodings.rst` |
| Random datasets | `tutorials/how_to_generate_a_random_repertoire_dataset.rst` |
| Clustering | `tutorials/clustering_tutorial.rst` |
| Generative models | `tutorials/how_to_train_and_apply_gen_model.rst` |
| LIgO simulation | `quickstart/simulation_quickstart.rst`, `tutorials/ligo_simulation_yaml.rst`, `tutorials/how_to_simulate_*.rst`, `tutorials/simulation_with_custom_signal_functions.rst` |
| Motif discovery | `tutorials/discover_motifs_*.rst`, `tutorials/motif_recovery.rst` |
| ML methodology (nested cross-validation) | `tutorials/how_to_properly_train_and_assess_an_ml_model.rst`, `implementation_details/` |
| Developer guides | `developer_docs/` |
| Standalone example spec | `_static/files/quickstart/cli/quickstart.yaml` |

`usecases/*` and `_static/files/airrc_tutorial/` are old, so treat their YAML as legacy and ignore their errors.
