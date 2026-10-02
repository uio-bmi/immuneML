Dataset simulation with LIgO
==================================================================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: LIgO: simulate antigen or disease-associated signals in AIRR datasets
   :twitter:description: See tutorials on how to simulate antigen or disease-associated signals in AIRR datasets.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


For simulation of AIRR datasets with user-defined signals, immuneML uses LIgO. It supports simulation on both
repertoire and receptor level. It can be used from immuneML through LigoSim instruction.

For more details on the decisions behind simulation, see the Methods section of the
original paper:

Chernigovskaya, M., Pavlović, M., Kanduri, C., et al. (2025). Simulation of adaptive immune receptors
and repertoires with complex immune information to guide the development and benchmarking of AIRR machine learning.
Nucleic Acids Research, 53(3), gkaf025. https://doi.org/10.1093/nar/gkaf025


See LIgO tutorials below:

.. toctree::
  :maxdepth: 1

  ligo_simulation_yaml
  how_to_simulate_co-occuring_signals
  how_to_simulate_paired_chain_data
  simulation_with_custom_signal_functions
