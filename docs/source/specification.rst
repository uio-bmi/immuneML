Specification
#############

.. toctree::
   :maxdepth: 2

The YAML specification defines which analysis should be performed by immuneML. It consists of two parts:

  - definitions,
  - instructions and
  - output.

Definitions describe datasets, encodings, ML methods, preprocessing, simulations and other components (see details below).
Instructions describe the analysis that will be performed and use datasets, encodings and other definitions to define on which data the
analysis will be performed and what are the specific parameters of that analysis. Output defines how to format the presentation of the
results of the analysis.

The overall structure of the YAML specification is the following:

.. indent with spaces
.. code-block:: yaml

  definitions: # mandatory keyword
    datasets: # mandatory keyword
      my_dataset_1: # user-defined name of the dataset
        ... # see below for the specification of the dataset
    encodings: # optional keyword - present if encodings are used
      my_encoding_1: # user-defined name of the encoding
        ... # see below for the specification of different encodings
    ml_methods: # optional keyword - present if ML methods are used
      my_ml_method_1: # user-defined name of the ML method
        ... # see below for the specification of different ML methods
    preprocessing_sequences: # optional keyword - present if preprocessing sequences are used
      my_preprocessing: # user-defined name of the preprocessing sequence
        ... # see below for the specification of different preprocessing
    reports: # optional keyword - present if reports are used
      my_report_1:
        ... # see below for the specification of different reports
  instructions: # mandatory keyword - at least one instruction has to be specified
    my_instruction_1: # user-defined name of the instruction
      ... # see below for the specification of different instructions
  output: # how to present the result after running (the only valid option now)
    format: HTML

The logic behind parsing this specification is the following: anything defined under `definitions` is available in the `instructions` part, but
anything generated from the instructions is not available to other instructions. If output of one instruction (e.g. a generated dataset) needs to be
used in the other instruction, these two instructions have to be two separate analyses. In the second instruction the generated dataset for instance,
would then be defined under `definitions`/`datasets` section.

For details on each of these components, see the documentation below.

Definitions
===========

Datasets
--------

.. include:: ../specs/definitions/datasets.rst

Simulation
----------

.. include:: ../specs/definitions/simulation.rst

Encodings
---------

.. include:: ../specs/definitions/encodings.rst

Reports
---------

.. include:: ../specs/definitions/reports.rst

ML methods
-----------

.. include:: ../specs/definitions/ml_methods.rst

Preprocessings
--------------

.. include:: ../specs/definitions/preprocessings.rst

Instructions
============

DatasetGeneration
------------------

.. include:: ../specs/instructions/dataset_generation.rst

ExploratoryAnalysis
--------------------

.. include:: ../specs/instructions/exploratory_analysis.rst

HPOptimization
--------------------

.. include:: ../specs/instructions/hp.rst

Simulation
--------------------

.. include:: ../specs/instructions/simulation.rst

Output
======

HTML
----

.. include:: ../specs/output/outputs.rst
