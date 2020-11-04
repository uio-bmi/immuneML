How to specify an analysis with YAML
====================================

The domain-specific language (DSL) developed for immuneML defines what the analysis YAML specification should look like.
Depending on the specification, immuneML can execute different tasks (perform hyperparameter optimization, perform exploratory
analyses or make simulated immune receptor datasets (i.e., implant signals [e.g., k-mers] into existing AIRRe datasets). For a full overview of
all the options that can be specified, see :ref:`YAML specification`.
An example of the full YAML specification for training an ML model through nested cross-validation is given below.

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      d1:
        format: AdaptiveBiotech
        params:
          metadata_file: metadata.csv
          path: path_to_data_folder/
          result_path: path_to_result_folder/
    encodings:
      e1:
        KmerFrequency:
          k: 3
      e2:
        Word2Vec:
          vector_size: 16
          model_type: sequence
    ml_methods:
      log_reg1:
        SimpleLogisticRegression:
            C: 0.001
    reports:
      r1: SequenceLengthDistribution
    preprocessing_sequences:
      seq1:
        - filter_chain_B:
          ChainRepertoireFilter:
            keep_chain: A
  instructions:
    my_instruction1:
      type: TrainMLModel
      settings:
        -   preprocessing: seq1
            encoding: e1
            ml_method: log_reg1
        -   encoding: e2
            ml_method: log_reg1
      assessment:
        split_strategy: random
        split_count: 1
        training_percentage: 70
      selection:
        split_strategy: k_fold
        split_count: 5
        reports:
          data_splits: [r1]
      labels: [CD]
      dataset: d1
      strategy: GridSearch
      metrics: [accuracy, f1_micro]
      optimization_metric: accuracy
      reports: [r1]
      refit_optimal_model: False
      store_encoded_data: False
  output:
    format: HTML

Structure of the analysis specification
---------------------------------------

The analysis specification consists of three main parts: definitions, instructions and output.

Specifying Definitions
^^^^^^^^^^^^^^^^^^^^^^

Definitions refer to components, which will be used within the instructions. They include:

- Datasets definitions: specifying where data is located and how it should be imported,

- Preprocessing sequences: defining preprocessing steps to be taken on the dataset,

- Encodings: different data representation techniques,

- ML methods: different machine learning methods (e.g., SVM or KNN),

- Reports: specific analysis to be run on different parts of the data, ML methods or results.

Simulation-specific components (only relevant when running a :ref:`Simulation instruction<How to simulate antigen/disease-associated signals in AIRR datasets>`) are:

- Motifs: parts of the simulation definition defined by a seed and a way to create specific motif instances from the seed,

- Signals: parts of the :ref:`simulation<How to simulate antigen/disease-associated signals in AIRR datasets>` which can include multiple motifs and correspond to a single label for subsequent classification tasks,

- :ref:`Simulations<How to simulate antigen/disease-associated signals in AIRR datasets>`: define how to combine different signals and how to implant them in the dataset.

Each component is defined using a key (a string) that uniquely identifies it and which
will be used in the instructions to refer to the component defined in this way.
For example, a dataset used in Emerson et al. 2017 (Nature Genetics), may  be defined
as follows:

.. highlight:: yaml
.. code-block:: yaml

  Emerson2017_dataset: # user-defined key (dataset name)
    format: AdaptiveBiotech
    params:
      path: ./Emerson2017/
      result_path: ./Emerson2017_immuneML/

Each definition component (listed above) is defined under its own key.
All component sections are located under **definitions** in the YAML specification file.
An example of sections with defined components is given below:

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    datasets:
      Emerson2017_dataset:
        format: AdaptiveBiotech
        params:
          path: ./Emerson2017/
          result_path: ./Emerson2017_immuneML/
    encodings:
      kmer_freq_encoding: KmerFrequency
    ml_methods:
      log_reg: LogisticRegression
    preprocessing_sequences:
      beta_chain_filter:
        - ChainRepertoireFilter:
            keep_chain: B
    reports:
      seq_length_distribution: SequenceLengthDistribution
    motifs:
      simple_motif:
      seed: AAA
      instantiation: GappedKmer
    signals:
      simple_signal:
        motifs:
          - simple_motif
        implanting: HealthySequence
    simulation:
      my_simulation:
        my_implanting:
          signals:
            - simple_signal
          dataset_implanting_rate: 0.5
          repertoire_implanting_rate: 0.1

Specifying Instructions
^^^^^^^^^^^^^^^^^^^^^^^

Instructions are defined similarly  to components: a key represents an identifier of
the instruction and type denotes the instruction that will be performed. The components,
which were defined previously will be used here as input to instructions.
The parameters for the instructions depend on the type of the instruction.
Instruction YAML specifications are located under **instructions** in the YAML specification file.

Some of the possible instructions are (see ):

- Training an ML model

- Exploratory analysis

- Simulation

Anything defined under definitions is available in the instructions part, but anything generated from the instructions is not available to other
instructions. If the output of one instruction needs to be used in another other instruction, two separate immuneML runs need to be made (e.g,
running immuneML once with the Simulation instruction to generate a dataset, and subsequently using that dataset as an input to a second immuneML
run to train a ML model).

An example of the YAML specification for the Training a ML model instruction is as follows:

.. highlight:: yaml
.. code-block:: yaml

  my_instruction: # user-defined instruction key
    type: TrainMLModel
    settings:
    - preprocessing: None
      encoding: kmer_freq_encoding
      ml_method: log_reg
    - preprocessing: beta_chain_filter
      encoding: kmer_freq_encoding
      ml_method: log_reg
    assessment:
      split_strategy: random
      split_count: 1
      training_percentage: 70
      reports:
        data_splits: [seq_length_distribution]
    selection:
      split_strategy: k_fold
      split_count: 5
    labels: [CMV]
    dataset: Emerson2017_dataset
    strategy: GridSearch
    metrics: [accuracy]
    optimization_metric: accuracy
    reports: []
    refit_optimal_model: False
    store_encoded_data: False

Output - HTML
^^^^^^^^^^^^^

The output section of the YAML specification defines the summary output of the execution of
immuneML. Currently, only HTML output format is supported. An index.html file will be created with links to a separate HTML file for each
instruction that was listed in the YAML specification. The instruction HTML pages will
include an overview of the instruction parameters (e.g., information on the dataset,
number of examples (number of repertoires or receptors), type of the dataset, the performance and ML model details of the nested cross-validation,
metrics used) and results (overview of performance results in the nested cross-validation loops,
outputs of individual reports). At this point, the HTML output is not customizable.

Running the specified analysis
------------------------------

To run an instruction via command line with the given YAML specification file:

.. code-block:: console

  immune-ml path/to/specification.yaml result/folder/path/

Alternatively, create an ImmuneMLApp object in a Python script and pass it the path parameter to the constructor before calling its `run()` method as follows:

.. highlight:: python
.. code-block:: python

  from source.app.ImmuneMLApp import ImmuneMLApp

  app = ImmuneMLApp(specification_path="path/to/specification.yaml", result_path="result/folder/path/")
  app.run()
