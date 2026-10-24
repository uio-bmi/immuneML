# Writing immuneML YAML specs

Paths are relative to `immuneML/`. Parameters and defaults are not listed here because they change. For an instruction, read its docstring (`workflows/instructions/**/<Type>Instruction.py`), its parser (`dsl/instruction_parsers/<Type>Parser.py`, which is where the actual validation happens) and its defaults (`config/default_params/instructions/<snake_type>_params.yaml`).

## Parsing rules (`dsl/`)
- **Component syntax:**
  - with parameters: `name: {ClassName: {params}}`, a single-key dict;
  - with no parameters: `name: ClassName`;
  - datasets are different: `name: {format: <X>, params: {...}}`.
- **Parameter merging:** parameters are merged over `config/default_params/<kind>/<snake_case>_params.yaml`. An unknown parameter is an error.
- **Names:** keys may contain only `[A-Za-z0-9_]`. All definitions share **one namespace**, so a duplicate name silently overwrites the earlier definition.
- **Instruction `type`:**
  - It is case-sensitive and equals the parser class name without `Parser`: TrainMLModel, MLApplication, ExploratoryAnalysis, Clustering, ValidateClustering, DatasetExport, SplitDataset, Subsampling, TrainGenModel, ApplyGenModel, LigoSim, FeasibilitySummary.
- **Instruction defaults are merged shallowly.** A nested dict you provide replaces the default dict entirely, so give every key in it.
- **Most parsers require every key.** They use `ParameterValidator.assert_keys(..., exclusive=True)`, so any key without a default must be written out. `null` fails unless the validator allows it.
- **Check the applied defaults:** every run writes `<out>/full_<spec>.yaml` with all defaults filled in.
- **Running:** `immune-ml spec.yaml out_dir`. If `out_dir` is non-empty, a timestamped folder is used instead. Results are browsable from `<out>/index.html`.

## Skeleton
```yaml
definitions:
  datasets:
    d1: {format: AIRR, params: {path: data/, metadata_file: metadata.csv}}   # repertoires need metadata_file
  encodings:
    e1: {KmerFrequency: {k: 3}}
  ml_methods:
    lr: LogisticRegression
  reports:
    r1: SequenceLengthDistribution
instructions:
  inst1: {type: <Type>, ...}
output: {format: HTML}
```

## Datasets
- Dataset type:
  - `is_repertoire: true` gives a RepertoireDataset and needs `metadata_file`. The metadata is a CSV with a `filename` column plus one column per label.
  - Otherwise you get a Sequence dataset, or a Receptor dataset when `paired: true`.
  - Receptor datasets need `receptor_chains` (e.g. `TRA_TRB`) and a `cell_id` column.
- Labels:
  - repertoire datasets: the metadata columns;
  - sequence/receptor datasets: non-AIRR columns, optionally restricted with `label_columns`.
- Re-importing a dataset that immuneML exported: AIRR format with `dataset_file: <exported>.yaml`.
- For test or demo data, use the `RandomRepertoireDataset`, `RandomSequenceDataset` or `RandomReceptorDataset` formats.

## Instructions: structure and gotchas

### TrainMLModel
Nested cross-validation for classifiers.
- **Required keys:** `dataset`, `labels`, `settings`, `optimization_metric`.
- **`settings`:** a list of `{encoding, ml_method, preprocessing?, dim_reduction?}`. Preprocessing used in a setting must not change the number of examples.
- **`labels`:** a list of `name` or `{name: {positive_class: X}}`.
- **`assessment` / `selection`:** each takes a SplitConfig (`hyperparameter_optimization/config/SplitConfig.py`).
  - `split_strategy` is one of: `random`, `k_fold`, `stratified_k_fold`, `loocv`, `manual`, `leave_one_out_stratification`.
  - `stratified_k_fold` needs exactly one label. It can be stratified by another metadata field with `stratified_k_fold_config: {stratification_label: <field>}`.
- **Report placement:**
  - top-level `reports:` accepts **only TrainMLModelReports**;
  - other reports go under `assessment`/`selection` → `reports: {data: [...], data_splits: [...], encoding: [...], models: [...]}`.
- **Hyperparameter grid for one method:** give list-valued parameters plus sibling keys:
  `lr: {LogisticRegression: {C: [0.1, 1]}, model_selection_cv: True, model_selection_n_folds: 5}`.
- **Metrics:** lower-case names from `ml_metrics/ClassificationMetric.py`.
- **Outputs:** the trained model zip in `<out>/<inst>/optimal_<label>/`, which is used by MLApplication.

### MLApplication
- Keys: `dataset`, `config_path` (the .zip from TrainMLModel), and optionally `metrics` and `device` (cpu/cuda/mps).
- Output: `predictions.csv`.

### ExploratoryAnalysis
- Keys: `analyses: {name: {dataset, reports, encoding?, labels?, preprocessing_sequence?, dim_reduction?}}`.
- Without an encoding, use DataReports; with an encoding, use EncodingReports.
- `dim_reduction` works only with KmerFrequency or Word2Vec encodings.

### Clustering
- Required keys: `dataset`, `metrics`, `clustering_settings`, a list of `{encoding, method, dim_reduction?}`.
- **Always give the full `sample_config` and `stability_config`, each with an int `random_seed`.** The default null seed fails validation.
- Metrics:
  - internal: silhouette, calinski_harabasz, davies_bouldin;
  - external: ARI, AMI, etc. These need `labels`.
- Reports: ClusteringReport, ClusteringMethodReport, EncodingReport.

### ValidateClustering
- Keys: `clustering_config_path` (the zip exported by Clustering), `dataset`, `metrics`, `validation_type: [method_based, result_based]`.

### SplitDataset
- `split_config` takes a `random` or `manual` strategy with `split_count: 1`, or `group_k_fold` with `split_count` ≥ 2 and a `group_k_fold_config` (see the SplitConfig docstring).
- One split is exported to `<inst>/train/` and `<inst>/test/`; group_k_fold exports `<inst>/split_<i>/train/` and `test/` plus `group_assignment.csv`.
- It is typically used to create discovery and validation sets for clustering.

### Subsampling
- Required: `subsampled_dataset_sizes`.
- `label` and `subsampled_class_distributions` must be given together.

### DatasetExport
- Keys: `datasets: [...]`, and optionally `preprocessing_sequence`.
- It always exports AIRR; there is no `export_formats`.

### TrainGenModel
- The dataset must be a Sequence or Receptor dataset; Repertoire datasets are not accepted.
- Give exactly one of `method` or `methods`.
- OLGA and ExperimentalImport are not trainable.
- Output: a model zip used by ApplyGenModel.

### ApplyGenModel
- All keys are required: `gen_examples_count`, `ml_config_path`, `reports` (may be `[]`).

### LigoSim
- **`export_p_gens` is required.**
- `simulation` refers to `definitions/simulations`.

### FeasibilitySummary
- Takes `simulation`, `sequence_count` and `number_of_processes`.

## LIgO simulation definitions (`simulation/`, `dsl/definition_parsers/{Motif,Signal,Simulation}Parser.py`)

Structure:
```yaml
definitions:
  motifs:  {m1: {seed: AS}}                       # seed motif (or PWM: {file_path, threshold})
  signals: {s1: {motifs: [m1]}}                   # name must not contain "__"
  simulations:
    sim1:
      is_repertoire: true
      paired: false
      sequence_type: amino_acid
      simulation_strategy: RejectionSampling      # or Implanting; required
      sim_items:
        item1:
          generative_model: {type: OLGA, default_model_name: humanTRB}
          number_of_examples: 10
          receptors_in_repertoire_count: 6        # repertoire simulations only
          signals: {s1: 0.5}                      # "s1__s2" means co-occurring signals
instructions:
  inst1: {type: LigoSim, simulation: sim1, export_p_gens: false}
```

- Only `OLGA` and `ExperimentalImport` can be generative models in LIgO.
- OLGA:
  - It has no `chain` key. Use `default_model_name` and/or `locus`.
  - Write `model_path: null`, not `None`.
- Implanting does not support signal pairs, custom signal functions, or V/J-gene conditions. Custom signal functions (`source_file` + `is_present_func`) need RejectionSampling.
- `sequence_len_limits` needs both `min` and `max`.

## Generative models and clustering methods
- Generative models live in `ml_methods/generative_models/` (PWM, SimpleLSTM, SimpleVAE, SoNNia, ProGen, …).
  - ProGen's defaults file never loads because of a file-name mismatch, so give all of its parameters.
- Clustering methods (`ml_methods/clustering/`) and dimensionality reduction methods (`ml_methods/dim_reduction/`) pass their parameters through to sklearn/umap.
  - DBSCAN, HDBSCAN and AgglomerativeClustering cannot `predict`.
  - TSNE can only `fit_transform`.
