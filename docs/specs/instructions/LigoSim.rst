


LIgO simulation instruction creates a synthetic dataset from scratch based on the generative model and a set of signals provided by
the user.

**Specification arguments:**

- simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how to combine signals with simulated data; specified under definitions

- sequence_batch_size (int): how many sequences to generate at once using the generative model before checking for signals and filtering

- max_iterations (int): how many iterations are allowed when creating sequences

- export_p_gens (bool): whether to compute generation probabilities (if supported by the generative model) for sequences and include them as part of output

- number_of_processes (int): determines how many simulation items can be simulated in parallel


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_simulation_instruction: # user-defined name of the instruction
            type: LIgOSim # which instruction to execute
            simulation: sim1
            sequence_batch_size: 1000
            max_iterations: 1000
            export_p_gens: False
            number_of_processes: 4


