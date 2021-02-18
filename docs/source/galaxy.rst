immuneML & Galaxy
=================

All of immuneMLs functionalities are also available through `a Galaxy web interface <https://galaxy.immuneml.uio.no>`_ as a collection of Galaxy tools. We provide a YAML-based Galaxy
tool that is equivalent to the CLI (command-line interface), as well as repertoire and receptor-level classification tools with an intuitive
graphical user interface aimed at immunology experts without a machine learning background.

If you are unfamiliar with Galaxy, you may want to start here:

- :ref:`Introduction to Galaxy`

To get started using immuneML in Galaxy, you will need to add your dataset to Galaxy, which is explained in this tutorial:

- :ref:`How to make an immuneML dataset in Galaxy`

Remote datasets may be fetched from VDJdb or the iReceptor Plus Gateway, see:

- :ref:`How to import remote AIRR datasets in Galaxy`

If you do not want to use experimental data and just want to try something out quickly, you can simulate an immune dataset:

- :ref:`How to simulate an AIRR dataset in Galaxy`

Synthetic immune signals (representing antigen binding or disease) can be implanted in an existing dataset:

- :ref:`How to simulate immune events into an existing AIRR dataset in Galaxy`

Once an immuneML dataset has been created in Galaxy, immunology experts without machine learning background can follow these instructions:

- :ref:`How to train immune repertoire classifiers using the simplified Galaxy interface`

- :ref:`How to train immune receptor classifiers using the simplified Galaxy interface`


Alternatively, CLI equivalent tools based on the YAML specification can be run using the following instructions:

- :ref:`How to train ML models in Galaxy`

- :ref:`How to apply previously trained ML models to a new AIRR dataset in Galaxy`

- :ref:`How to run any AIRR ML analysis in Galaxy`

.. toctree::
  :maxdepth: 1
  :caption: Galaxy tutorials:

  galaxy/galaxy_intro

.. toctree::
  :maxdepth: 1
  :caption: Dataset tutorials:

  galaxy/galaxy_dataset
  galaxy/galaxy_import_remote_data
  galaxy/galaxy_simulate_dataset
  galaxy/galaxy_simulate_signals

.. toctree::
  :maxdepth: 1
  :caption: Analysis tutorials:

  galaxy/galaxy_simple_repertoires
  galaxy/galaxy_simple_receptors
  galaxy/galaxy_train_ml_models
  galaxy/galaxy_apply_ml_models
  galaxy/galaxy_general_yaml
