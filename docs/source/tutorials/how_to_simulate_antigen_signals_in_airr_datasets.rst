How to simulate antigen or disease-associated signals in AIRR datasets
======================================================================

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


YAML specification of the LigoSim instruction for introducing immune signals
---------------------------------------------------------------------------------

The YAML definition consists of three components: motif, signal and simulation definitions.

- :code:`motifs`: defined by either a (gapped) k-mer (see: :ref:`SeedMotif`) and how it might vary or by a
  position weight matrix (see: :ref:`PWM`).

- :code:`signals` (see: :ref:`Signal`): defined as a union of a set of motifs and AIR-specific information, such
  as V or J gene or IMGT position of the motif in the CDR3 sequence. Immune signals correspond to e.g., antigen-specificity.

- :code:`immune events`: immune events are sets of immune signals and their proportion in an AIRR. They correspond to
  diseases, vaccination, allergies. In practice, we define which signals are present and how often in a set of examples,
  and the immune event is assigned as a label for that example set.

- :code:`simulations` defines how the signals will be combined and simulated in the receptors or repertoires.

Simulation (as defined in :ref:`Simulation config` in the specification) groups examples (receptors or repertoires,
depending on the level of simulation desired by the user) with the same characteristics into simulation items
(:ref:`Simulation config item`) that precisely defines how this set of examples should be simulated.

Each simulation item defines the following for a set of examples:

- which signals should exist in that set of examples (and if it's repertoire-level simulation: in which percentage of
  each individual repertoire)

- what is the generative model that will create background AIR sequences which will be used as a starting point for
  simulation. Currently supported generative models for this purpose are OLGA (which can generate sequences from either
  one of the default OLGA models or from a custom model) and ExperimentalImport (which allows any set of sequences to
  be imported and used as background).

- immune events are defined on this level and have the same value for all examples within the given set.

See also the tutorial about :ref:`recovering simulated immune signals <Recovering simulated immune signals>`.

An example of a simulation with disease-associated signals is given below.

.. highlight:: yaml
.. code-block:: yaml

  definitions:
    motifs:
      motif1:
        seed: AA
      motif2:
        seed: GG
    signals:
      signal1:
        motifs: [motif1]
      signal2:
        motifs: [motif2]
    simulations:
      sim1:
        is_repertoire: true # if the simulation is on repertoire or receptor here -> here it's repertoire level
        paired: false # whether to simulate paired chain data or not
        sequence_type: amino_acid
        simulation_strategy: Implanting # how to simulate the signals
        remove_seqs_with_signals: true # remove signal-specific AIRs from the background
        sim_items:
          sim_item: # group of AIRs with the same parameters
            AIRR1:
              immune_events: # all repertoires in this set will have these values for immune events
                ievent1: True
                ievent1: False
              signals:
                signal1: 0.3 # in each repertoire 30% of sequences will have signal1
                signal2: 0.3 # in each repertoire other 30% of sequences will have signal2
              number_of_examples: 10 # simulate 10 repertoires
              receptors_in_repertoire_count: 6 # how many receptor sequences should be in each repertoire
              generative_model: # how to generate background AIRs
                chain: heavy
                default_model_name: humanIGH # use default model
                type: OLGA # use OLGA for background simulation
            AIRR2: # another set of repertoires, but with different parameters
              immune_events:
                ievent1: False
                ievent1: True
              signals: {signal1: 0.5, signal2: 0.5}
              number_of_examples: 10
              receptors_in_repertoire_count: 6
              generative_model:
                chain: heavy
                default_model_name: humanIGH
                model_path: null # if there was a custom model to use, path to the folder should be given here
                type: OLGA
  instructions:
    my_sim_inst:
      export_p_gens: false
      max_iterations: 100
      number_of_processes: 4
      sequence_batch_size: 1000
      simulation: sim1
      type: LigoSim
