Simulation with custom signal functions
==========================================

In LIgO, signals are most often defined using (gapped) k-mers or positional weight matrices. However, as this way of
defining signals can be limiting to a certain degree, LIgO also allows users to have more generic definition of a signal.
Signals can be specified using a **custom function** that takes a receptor sequence as input and outputs True/False
depending on whether the sequence satisfy some custom criteria from that function.

In this tutorial, we provide a simple example of what such functions might look like and how to specify the simulation
using signals with custom function. We assume that LIgO is already installed.

.. note::

    This way of defining signals can only be used in combination with rejection sampling.

Step 1: Defining the custom signal function
---------------------------------------------

The custom signal function is defined in a separate Python file. The function will always get the following arguments:

- amino acid sequence (sequence_aa)
- nucleotide sequence (sequence)
- V gene (v_call)
- J gene (j_call)

The output of the function should always be a single value, either True or False.

As the first step of the tutorial, save the following code to the file custom_function.py:

.. code-block:: python

    def is_present(sequence_aa: str, sequence: str, v_call: str, j_call: str) -> bool:
        return any(aa in sequence_aa for aa in ['A', 'T']) and len(sequence_aa) > 12

In this case, we assume that the sequence contains signal if it contains `A` or `T` and is longer than 12 amino acids.
In principle, any logic could be implemented inside this function.

Step 2: Define the YAML specification
-----------------------------------------

The YAML specification fully defines the simulation to be performed. Save the following specification to specs.yaml in
the same folder as custom_function.py.

Signal definition here (for signal1) consists of two parameters:

- is_present_func: stating the name of the function to use from the provided python file to assess if the signal is present,
- source_file: stating the path to the python file where the custom function is located.

The rest of the simulation is defined in the same way as when k-mers or PWMs are used.

.. code-block:: yaml

    definitions:
      signals:
        signal1: # signal with the custom signal function
          is_present_func: is_present
          source_file: custom_function.py
      simulations:
        sim1:
          is_repertoire: true
          paired: false
          sequence_type: amino_acid
          sim_items:
            sim_item1:
              generative_model: # use OLGA humanTRB model to generate sequence
                default_model_name: humanTRB
                type: OLGA
              number_of_examples: 10 # generate 10 repertoires
              receptors_in_repertoire_count: 6 # each repertoire should have 6 receptor sequences
              seed: 100 # random seed for sequence generation to ensure reproducibility
              signals: # which signals should be in the simulated repertoires
                signal1: 0.5 # 50% of receptor sequences should have signal1 and the rest should have no signal
          simulation_strategy: RejectionSampling # use rejection sampling to filter out sequences based on signal presence/absence
    instructions:
      inst1:
        export_p_gens: false # do not compute p gens for generated sequences
        max_iterations: 100
        number_of_processes: 1
        sequence_batch_size: 100
        simulation: sim1
        type: LigoSim
    output:
      format: HTML

Step 3: Running the simulation
----------------------------------

When the two files mentioned above are saved, run the following:

.. code-block:: console

    ligo specs.yaml simulation_output

The simulation for this specification should only take a few seconds and all results will be stored in the
`simulation_output` folder.



