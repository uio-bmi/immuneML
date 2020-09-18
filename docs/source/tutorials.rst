Tutorials
==========

This page provides an overview of tutorials on how to get started using immuneML and instructions on how to use immuneML for various different use cases.


.. toctree::
  :maxdepth: 1
  :caption: Tutorials:

All immuneML analyses are specified using a YAML specification file. To learn how to construct this file, see this tutorial:

  tutorials/how_to_specify_an_analysis_with_yaml

Each analysis begins with selecting the dataset that will be used. In immuneML, the user can choose to import an existing dataset, or to generate a
dataset made of random sequences (for example to test out some functionality without needing to use a specific dataset, or as a benchmarking dataset).
The respective tutorials can be found here:

  tutorials/how_to_import_the_data_to_immuneML
  tutorials/how_to_generate_a_random_repertoire_dataset

Using the specified dataset, immuneML can be used for various purposes: one can train and assess an ML model for immune repertoire or receptor-level
classification, perform an exploratory analysis (to run preprocessings, encodings and reports without training a ML model), or simulate immune events
by implanting sequence motifs in the dataset. See the tutorials below:

  tutorials/how_to_train_and_assess_a_receptor_or_repertoire_classifier
  tutorials/how_to_perform_exploratory_analysis
  tutorials/how_to_simulate_antigen_signals_in_airr_datasets