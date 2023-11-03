How to simulate co-occuring immune signals
---------------------------------------------------------------------

LIgO supports simulation of co-occurring immune signals using rejection sampling. In this tutorial, we replicate the simulation of the usecase 2 from the LIgO manuscript. Briefly, we will perform repertoire-level simulation, where some TRBs contain two signals belonging to two different immune events. We define signal 1 as a 3-mer GDT and signal 2 as a 3-mer SGL.

Step 1: Define immune signals
````````````````````````````````````

We begin by defining immune signals for simulation. This step remains consistent with standard LIgO simulation, even when we aim to simulate the occurrence of two immune signals within one receptor.


.. code-block:: yaml

  definitions:
    motifs:
      motif1:
        seed: GDT
      motif2:
        seed: SGL
    signals:
      signal1:
        motifs: [motif1]
      signal2:
        motifs: [motif2]

Step 2: Define frequency of each individual signal and the pair of signals in a repertoire
````````````````````````````````````````````````````````````````````````
.. code-block:: yaml

  simulations:
      sim1:
        is_repertoire: true
        paired: false
        sequence_type: amino_acid
        simulation_strategy: RejectionSampling
        sim_items:
          AIRR1:
            generative_model:
              chain: beta
              default_model_name: humanTRB
              model_path: null
              type: OLGA
            is_noise: false
            number_of_examples: 10 # we simulate 10 reprtoires
            receptors_in_repertoire_count: 1000 # we simulate 1000 BCRs in each repertoire
            signals:
              signal1__signal2: 0.1 # 10% of BCRs contain both signal 1 and signal 2
              signal1: 0.2 # 20% of BCRs contain signal 1 
              signal2: 0.2 # 20% of BCRs contain signal 2


Step 3: Run the simulation with the following yaml file 
`````````````````````````````

.. code-block:: yaml

  definitions:
    motifs:
      motif1:
        seed: GDT
      motif2:
        seed: SGL
    signals:
      signal1:
        motifs: [motif1]
      signal2:
        motifs: [motif2]
  simulations:
      sim1:
        is_repertoire: true
        paired: false
        sequence_type: amino_acid
        simulation_strategy: RejectionSampling
        sim_items:
          AIRR1:
            generative_model:
              chain: beta
              default_model_name: humanTRB
              model_path: null
              type: OLGA
            is_noise: false
            number_of_examples: 10 # we simulate 10 reprtoires
            receptors_in_repertoire_count: 1000 # we simulate 1000 BCRs in each repertoire
            signals:
              signal1__signal2: 0.1 # 10% of BCRs contain both signal 1 and signal 2
              signal1: 0.2 # 20% of BCRs contain signal 1 
              signal2: 0.2 # 20% of BCRs contain signal 2
  instructions:
    inst1:
      export_p_gens: false # could take some time to compute (from olga)
      max_iterations: 1000
      number_of_processes: 4
      sequence_batch_size: 100000
      simulation: sim1
      type: LigoSim




