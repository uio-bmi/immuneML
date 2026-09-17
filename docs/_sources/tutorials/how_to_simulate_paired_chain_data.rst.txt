Paired chain simulations in LIgO
==================================

LIgO supports paired chain simulations under a simplifying assumption that the individual chains are generated independently, but allows for some logic
in terms of how to pair the chains.

This tutorial contains two sections: paired chain simulation on the receptor level and paired chain simulation on the
repertoire level.

Paired chain simulation on the receptor level
------------------------------------------------

To simulate paired receptor sequences, we define how to simulate each chain, and then how to pair them.

Step 1: Define immune signals
````````````````````````````````````

We first define immune signals that will be used for simulation. There are no additional restrictions on how to define them
compared to any other LIgO simulation, except that if the immune signal contains V or J gene, it can only be present in
the chain that has those genes (e.g., signal with TRBV1 can only exist in TRB sequences, but not in TRA).

We define two immune signals consisting of one k-mer each:

.. code-block:: yaml

      motifs:
        motif1:
          seed: AS
        motif2:
          seed: GG
      signals:
        signal1:
          motifs: [motif1]
        signal2:
          motifs: [motif2]

Step 2: Define the simulation of individual chains and their pairing
````````````````````````````````````````````````````````````````````````

The following specification defines the simulation with 10 TRA sequences and TRB sequences, and uses `paired` parameter
to specify which simulation items to combine (here `sim_alpha` and `sim_beta`).

.. code-block:: yaml

      simulations:
        sim1:
          sim_items:
            sim_alpha: # how to simulate alpha chain
              generative_model:
                default_model_name: humanTRA
                type: OLGA
              number_of_examples: 10 # 10 sequences
              seed: 100
              signals:
                signal2: 1 # all of the sequences should contain signal2
            sim_beta: # how to simulate beta chain
              generative_model:
                default_model_name: humanTRB
                type: OLGA
              number_of_examples: 10 # 10 sequences
              seed: 2
              signals:
                signal1: 1 # all of the sequences should contain signal1
          is_repertoire: false # simulation is on the receptor level, not repertoire
          paired: # how to pair simulation items
          - [sim_alpha, sim_beta] # here only one combination, but could be many
          sequence_type: amino_acid
          simulation_strategy: RejectionSampling

Step 3: Run the simulation
`````````````````````````````

Save the full specification to `paired_sim_receptor_specs.yaml`:

.. code-block:: yaml

    definitions:
      motifs:
        motif1:
          seed: AS
        motif2:
          seed: GG
      signals:
        signal1:
          motifs: [motif1]
        signal2:
          motifs: [motif2]
      simulations:
        sim1:
          sim_items:
            sim_alpha: # how to simulate alpha chain
              generative_model:
                default_model_name: humanTRA
                type: OLGA
              number_of_examples: 10 # 10 sequences
              seed: 100
              signals:
                signal2: 1 # all of the sequences should contain signal2
            sim_beta: # how to simulate beta chain
              generative_model:
                default_model_name: humanTRB
                type: OLGA
              number_of_examples: 10 # 10 sequences
              seed: 2
              signals:
                signal1: 1 # all of the sequences should contain signal1
          is_repertoire: false # simulation is on the receptor level, not repertoire
          paired: # how to pair simulation items
          - [sim_alpha, sim_beta] # here only one combination, but could be many
          sequence_type: amino_acid
          simulation_strategy: RejectionSampling
    instructions:
      inst1: # user-defined instruction name and the name of the output folder
        export_p_gens: false
        max_iterations: 100
        number_of_processes: 2
        sequence_batch_size: 100
        simulation: sim1
        type: LigoSim
    output:
      format: HTML

Run LIgO using the following command:

.. code-block:: console

    ligo paired_sim_receptor_specs.yaml simulation_output_receptor

All results will be located in `simulation_output_receptor` folder.

Step 4: Explore results of receptor-level simulation
```````````````````````````````````````````````````````

The simulated dataset is located under `simulation_output_receptor/inst1/exported_dataset/airr/batch1.tsv`. Some
of the columns are shown in the table below:

.. list-table:: Simulated receptors in AIRR format
    :header-rows: 1

    * - junction_aa
      - locus
      - cell_id
      - signal1
      - signal2
    * - CAFHGGATNKLIF
      - TRA
      - eb73d6fabc684aa5bb2c3faaff5bc1d1
      - 0
      - 1
    * - CASSESEKVRSSTDTQYF
      - TRB
      - eb73d6fabc684aa5bb2c3faaff5bc1d1
      - 1
      - 0
    * - CAETGGTSYGKLTF
      - TRA
      - 307e92e0fd734fc48239aaa8af911637
      - 0
      - 1
    * - CASSPEGQGCNQPQHF
      - TRB
      - 307e92e0fd734fc48239aaa8af911637
      - 1
      - 0

In the output, each row represent one chain, and if the chains come from the same receptor, they have the same `cell_id`.
The `cell_id` field contains a unique hex value to fully determine the cell.

Paired chain simulation on the repertoire level
------------------------------------------------

The paired chain simulation on the repertoire level is defined in the same way as the receptor level simulation. The
difference between these two levels comes from the fact that not all receptors in the repertoire contain signals, so
the signal pairing itself is slightly different.

Signals for repertoires are defined in the following way:

.. code-block:: yaml

    signals:
        signal1: 0.2
        signal2: 0.1

This means that 20% of the repertoire sequences will contain signal1 and 10% of the sequences will contain signal2.
This definition is on the level of a single chain for one repertoire (e.g., beta).

For the other chain (e.g., alpha), the repertoire signals could be defined like this:

.. code-block:: yaml

    signals:
        signal1: 0.1
        signal2: 0.2

This means that 10% of the repertoire sequences will contain signal1 and 20% will contain signal2.

If these were now to be used to make paired chain repertoire, the resulting repertoires will contain:

- 10% of receptors with signal1 in both chains,
- 10% of receptors in with signal1 in beta chain and signal2 in alpha chain,
- 10% of receptors with signal2 in both chains.

The chains are simply combined in the order that signals are defined.

If for the alpha chain for example, the proportion of signal2 was larger that 0.2, the remaining alpha sequences with
signal2 would be paired with beta chain sequences without any signal.

Step 1: Define the YAML specification for repertoire-level paired chain simulation
```````````````````````````````````````````````````````````````````````````````````````

Save the specification to `paired_sim_repertoire_specs.yaml`:

.. code-block:: yaml

    definitions:
      motifs:
        motif1:
          seed: AS
        motif2:
          seed: GG
      signals:
        signal1:
          motifs: [motif1]
        signal2:
          motifs: [motif2]
      simulations:
        sim1:
          is_repertoire: true
          paired:
            - [sim_alpha, sim_beta] # combine repertoires from these two simulation items
          sequence_type: amino_acid
          sim_items:
            sim_alpha:
              generative_model:
                default_model_name: humanTRA
                type: OLGA
              number_of_examples: 10 # 10 repertoires
              receptors_in_repertoire_count: 10 # 10 alpha chain sequences per repertoire
              seed: 100
              signals:
                signal1: 0.1
                signal2: 0.2
            sim_beta:
              generative_model:
                default_model_name: humanTRB
                type: OLGA
              number_of_examples: 10 # 10 repertoires
              receptors_in_repertoire_count: 10 # 10 beta chain sequences per repertoire
              seed: 2
              signals:
                signal1: 0.2
                signal2: 0.1
          simulation_strategy: RejectionSampling
    instructions:
      inst1:
        export_p_gens: false
        max_iterations: 100
        number_of_processes: 2
        sequence_batch_size: 100
        simulation: sim1
        type: LigoSim
    output:
      format: HTML

Step 2: Run the simulation
`````````````````````````````

Run LIgO using the following command:

.. code-block:: console

    ligo paired_sim_repertoire_specs.yaml simulation_output_repertoire

All results will be located in `simulation_output_repertoire` folder. Each repertoire file will contain alpha and beta
chain sequences. The sequences coming from the same receptor will have the same `cell_id` as illustrated in the receptor
simulation case above.

