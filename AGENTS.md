# AGENTS.md

This file is for AI coding agents working with immuneML. It explains how to write and run YAML analysis specs and how to add new components, without exploring the whole codebase first.

**How to use it**
1. Read this file, then open the detail file in `agentic_readme/` that fits your task.
2. For any concrete component, read its class docstring: it is the source of truth for what the component does and how to specify it.
   - Parameters are documented under `**Specification arguments:**` and `**YAML specification:**`.
   - Defaults are in `immuneML/config/default_params/`.
   - Instruction rules are enforced in `immuneML/dsl/instruction_parsers/<Type>Parser.py`.
   - If these notes and the code disagree, the code is right.
3. Docs can lag behind the code, and a local `docs/specs/` folder can be stale. When a tutorial and the code disagree, follow the code.

Galaxy is not covered. Paths are relative to the repository root.

## Detail files (`agentic_readme/`)

| File | Contents |
|---|---|
| `yaml_specs.md` | Parsing rules, a spec skeleton, and the structure and gotchas of each instruction, plus LIgO simulation, generative models and clustering |
| `compatibility.md` | Encoder × dataset type, classifier × encoder, report prerequisites, and which report types each instruction accepts |
| `extending.md` | Adding components (base classes, docstring standard), data model essentials, and test patterns |
| `docs_map.md` | Which docs page to read for each topic, and which docs are legacy |

## Big picture
- immuneML is a machine-learning platform for adaptive immune receptor repertoire (AIRR) data (BCR/TCR). Every analysis is a YAML spec.
- Run a spec with `immune-ml spec.yaml out_dir`. Each run also writes `full_<spec>.yaml`, which shows every default that was applied.
- A spec has three parts:
  - `definitions:` — named datasets, encodings, ml_methods, reports, preprocessing_sequences, motifs, signals, simulations;
  - `instructions:` — workflows that refer to those names: TrainMLModel, MLApplication, ExploratoryAnalysis, Clustering, ValidateClustering, DatasetExport, SplitDataset, Subsampling, TrainGenModel, ApplyGenModel, LigoSim, FeasibilitySummary;
  - `output: {format: HTML}`.

## Analysis principles: veridical data science
Follow the PCS framework (Yu and Kumbier, "Veridical data science", PNAS 117(8), 2020) when you design, run, report or extend immuneML analyses. A result is trustworthy only if it is:
- **Predictable**: it holds on data not used to build it. Judge models on held-out data (nested cross-validation in TrainMLModel; ValidateClustering on a separate validation set), and use simulated data with known ground truth (eg LIgO) where reasonable to check that a method recovers what it should.
- **Computable**: it is reproducible. Keep the spec, the random seeds and `full_<spec>.yaml` with every result.
- **Stable**: it holds under **reasonable** perturbations of the data and of your own judgment calls, such as preprocessing, encoding, method, hyperparameters, splits and seeds. Compare several settings rather than one, and use the stability tools that exist (e.g. `stability_config` in Clustering).

Document the judgment calls you made and why, report results that were unstable as well as the ones that held up, and do not present a single run as a finding. When you add a component, make it easy to perturb and test in this way: expose the choices as parameters and accept a random seed.

## Finding components
Components are discovered by reflection; there is no registry.
- The YAML name is the class name. Encoders, importers and instructions drop their suffix (`KmerFrequencyEncoder` → `KmerFrequency`).
- A class is found only if its file is named after the class and sits in the folder for its kind.
- To see what exists, list the folder. To use a component, read its class docstring.
- Defaults are in `immuneML/config/default_params/<kind>/<snake_case_name>_params.yaml`.

| YAML section | Folder (under `immuneML/`) |
|---|---|
| `definitions/datasets` → `format` | `IO/dataset_import/` (`<Format>Import.py`) |
| `definitions/encodings` | `encodings/<family>/` (`<Name>Encoder.py`) |
| `definitions/preprocessing_sequences` | `preprocessing/`, filters in `preprocessing/filters/` |
| `definitions/ml_methods` | `ml_methods/{classifiers,clustering,dim_reduction,generative_models}/` |
| `definitions/reports` | `reports/<category>_reports/` |
| `definitions/example_weightings` | `example_weighting/` |
| `definitions/motifs`, `signals`, `simulations` | `simulation/` |
| `instructions` → `type` | `workflows/instructions/` (`<Type>Instruction.py`) |

Other folders in `immuneML/`: `data_model/` (datasets, AIRR sequence sets, EncodedData), `dsl/` (YAML parsing), `hyperparameter_optimization/` (nested cross-validation, SplitConfig), `util/` (helpers, including ReflectionHandler and ParameterValidator). Tests are in `test/` and mirror the package. Docs are in `docs/source/`.

## Setup and tests
- Python 3.11 is the version CI uses. Install with `pip install -r requirements-lock.txt`, or `pip install -e ".[<extra>]"` for optional extras (`DL`, `gen_models`, `embeddings`, `TCRdist`, `ligo`, …).
- Run tests from the repository root with `PYTHONPATH=. python -m pytest test/<path>`. Run the tests for the area you changed, not the whole suite. Tests that use CompAIRR need the `compairr` binary to be built.
- When writing tests, follow the test patterns in `agentic_readme/extending.md` (test cache, temporary folder, test data).
- To check a spec end to end, run it on a `Random*Dataset` and open `<out>/full_<spec>.yaml` to see the defaults that were applied.

## Code conventions
- Each new component needs a defaults YAML in `immuneML/config/default_params/<kind>/` and a docstring that follows the documentation standard. The user docs are generated from these docstrings.
- Match the style of the surrounding code. Do not reformat or refactor unrelated code.
- An optional dependency must be imported so that a missing package does not break other components.

## Boundaries
- Work only in `immuneML/`, `test/`, `docs/source/` and `agentic_readme/`. Galaxy-related code and docs are out of scope.
- Do not edit generated output in `docs/specs/`.
- Do not copy YAML from the legacy docs listed in `docs_map.md`.
- Commit and push only when asked. The working branch is `development`; PRs go to `master`.

## Keeping this up to date
Edit this file and the files in `agentic_readme/` only when rules change, such as instruction structure, base classes or parser validation. Keep them short and addressed to agents, and do not add parameter or component lists.

## Writing and running analyses

Best practice is always to run a spec on either a random dataset or on a small real dataset to see if everything works.
See Random*Dataset and Subsampling instruction for how to make such datasets.

When deciding on details of an analysis (eg which kmers to use for simulation), and an overview of options
could be obtained by running preliminary analyses, use immuneML for the purpose and keep the specs and results for 
transparency and reproducibility.