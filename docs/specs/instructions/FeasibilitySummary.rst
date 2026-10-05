


FeasibilitySummary instruction creates a small synthetic dataset and reports summary metrics to show if the simulation with the given
parameters is feasible. The input parameters to this analysis are the name of the simulation
(the same that can be used with LigoSim instruction later if feasibility analysis looks acceptable), and the number of sequences to
simulate for estimating the feasibility.

The feasibility analysis is performed for each generative model separately as these could differ in the analyses that will be reported.

**Specification arguments:**

- simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how to combine signals with simulated data; specified under definitions

- sequence_count (int): how many sequences to generate to estimate feasibility (default value: 100 000)

- number_of_processes (int): for the parts of the analysis that are possible to parallelize, how many processes to use


**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    instructions:
        my_feasibility_summary: # user-defined name of the instruction
            type: FeasibilitySummary # which instruction to execute
            simulation: sim1
            sequence_count: 10000


