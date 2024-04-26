Dataset simulation with LIgO
==================================================================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: simulate antigen or disease-associated signals in AIRR datasets
   :twitter:description: See tutorials on how to simulate antigen or disease-associated signals in AIRR datasets.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


For simulation of AIRR datasets with user-defined signals, immuneML uses LIgO. It supports simulation on both
repertoire and receptor level. For more details on the decisions behind simulation, see the Methods section of the
original paper:

Chernigovskaya, M., et al. (2023). Simulation of adaptive immune receptors and repertoires with complex immune
information to guide the development and benchmarking of AIRR machine learning (p. 2023.10.20.562936).
bioRxiv. https://doi.org/10.1101/2023.10.20.562936



.. toctree::
  :maxdepth: 1

  ligo_simulation_yaml
  how_to_simulate_co-occuring_signals
  how_to_simulate_paired_chain_data
  simulation_with_custom_signal_functions
