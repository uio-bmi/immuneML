
Motifs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Motifs are the objects which are implanted into sequences during simulation.
They are defined under :code:`definitions/motifs`. There are several different motif types, each
having their own parameters.


SeedMotif
''''''''''''''''''''''''''''''''''''''''''''''''''''


Describes motifs by seed, possible gaps, allowed hamming distances, positions that can be changed and what they can be changed to.

**Specification arguments:**

- seed (str): An amino acid sequence that represents the basic motif seed. All implanted motifs correspond to the seed, or a modified version thereof, as specified in its instantiation strategy. If this argument is set, seed_chain1 and seed_chain2 arguments are not used.

- min_gap (int): The minimum gap length, in case the original seed contains a gap.

- max_gap (int): The maximum gap length, in case the original seed contains a gap.

- hamming_distance_probabilities (dict): The probability of modifying the given seed with each number of modifications. The keys represent the number of modifications (hamming distance) between the original seed and the implanted motif, and the values represent the probabilities for the respective number of modifications. For example {0: 0.7, 1: 0.3} means that 30% of the time one position will be modified, and the remaining 70% of the time the motif will remain unmodified with respect to the seed. The values of hamming_distance_probabilities must sum to 1.

- position_weights (dict): A dictionary containing the relative probabilities of choosing each position for hamming distance modification. The keys represent the position in the seed, where counting starts at 0. If the index of a gap is specified in position_weights, it will be removed. The values represent the relative probabilities for modifying each position when it gets selected for modification. For example {0: 0.6, 1: 0, 2: 0.4} means that when a sequence is selected for a modification (as specified in hamming_distance_probabilities), then 60% of the time the amino acid at index 0 is modified, and the remaining 40% of the time the amino acid at index 2. If the values of position_weights do not sum to 1, the remainder will be redistributed over all positions, including those not specified.

- alphabet_weights (dict): A dictionary describing the relative probabilities of choosing each amino acid for hamming distance modification. The keys of the dictionary represent the amino acids and the values are the relative probabilities for choosing this amino acid. If the values of alphabet_weights do not sum to 1, the remainder will be redistributed over all possible amino acids, including those not specified.

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        motifs:
            # examples for single chain receptor data
            my_simple_motif: # this will be the identifier of the motif
                seed: AAA # motif is always AAA
            my_gapped_motif:
                seed: AA/A # this motif can be AAA, AA_A, CAA, CA_A, DAA, DA_A, EAA, EA_A
                min_gap: 0
                max_gap: 1
                hamming_distance_probabilities: # it can have a max of 1 substitution
                    0: 0.7
                    1: 0.3
                position_weights: # note that index 2, the position of the gap, is excluded from position_weights
                    0: 1 # only first position can be changed
                    1: 0
                    3: 0
                alphabet_weights: # the first A can be replaced by C, D or E
                    C: 0.4
                    D: 0.4
                    E: 0.2



PWM
''''''''''''''''''''''''''''''''''''''''''''''''''''


Motifs defined by a positional weight matrix and using bionumpy's PWM internally.
For more details on bionumpy's implementation of PWM, as well as for supported formats,
see the documentation at https://bionumpy.github.io/bionumpy/tutorials/position_weight_matrix.html.

**Specification arguments:**

- file_path: path to the file where the PWM is stored

- threshold (float): when matching PWM to a sequence, this is the threshold to consider the sequence as containing the motif

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        motifs:
            my_custom_pwm: # this will be the identifier of the motif
                file_path: my_pwm_1.csv
                threshold: 2



Signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


A signal represents a collection of motifs, and optionally, position weights showing where one
of the motifs of the signal can occur in a sequence.
The signals are defined under :code:`definitions/signals`.

A signal is associated with a metadata label, which is assigned to a receptor or repertoire.
For example antigen-specific/disease-associated (receptor) or diseased (repertoire).

.. note:: IMGT positions

    To use sequence position weights, IMGT positions should be explicitly specified as strings, under quotation marks, to allow for all positions to be properly distinguished.

**Specification arguments:**

- motifs (list): A list of the motifs associated with this signal, either defined by seed or by position weight matrix. Alternatively, it can be a list of a list of motifs, in which case the motifs in the same sublist (max 2 motifs) have to co-occur in the same sequence

- sequence_position_weights (dict): a dictionary specifying for each IMGT position in the sequence how likely it is for the signal to be there. If the position is not present in the sequence, the probability of the signal occurring at that position will be redistributed to other positions with probabilities that are not explicitly set to 0 by the user.

- v_call (str): V gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

- j_call (str): J gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

- source_file (str): path to the file where the custom signal function is; cannot be combined with the arguments listed above (motifs, v_call, j_call, sequence_position_weights)

- is_present_func (str): name of the function from the source_file file that will be used to specify the signal; the function's signature must be:

.. code-block:: python

    def is_present(sequence_aa: str, sequence: str, v_call: str, j_call: str) -> bool:
        # custom implementation where all or some of these arguments can be used

- clonal_frequency (dict): clonal frequency in Ligo is simulated through `scipy's zeta distribution function for generating random numbers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zipf.html>`_, with parameters provided under clonal_frequency parameter. If clonal frequency should not be used, this parameter can be None

.. code-block:: yaml

  clonal_frequency:
    a: 2 # shape parameter of the distribution
    loc: 0 # 0 by default but can be used to shift the distribution


**YAML specification:**

.. code-block:: yaml

    definitions:
        signals:
            my_signal:
                motifs:
                    - my_simple_motif
                    - my_gapped_motif
                sequence_position_weights:
                    '109': 0.5
                    '110': 0.5
                v_call: TRBV1
                j_call: TRBJ1
                clonal_frequency:
                    a: 2
                    loc: 0
            signal_with_custom_func:
                source_file: signal_func.py
                is_present_func: is_signal_present
                clonal_frequency:
                    a: 2
                    loc: 0



Simulation config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The simulation config defines all parameters of the simulation.
It can contain one or more simulation config items, which define groups of repertoires or receptors
that have the same simulation parameters, such as signals, generative model, clonal frequencies, and noise parameters.


**Specification arguments:**

- sim_items (dict): a list of SimConfigItems defining individual units of simulation

- is_repertoire (bool): whether the simulation is on a repertoire (person) or sequence/receptor level

- paired: if the simulation should output paired data, this parameter should contain a list of a list of sim_item pairs referenced by name that should be combined; if paired data is not needed, then it should be False

- sequence_type (str): either amino_acid or nucleotide

- simulation_strategy (str): either RejectionSampling or Implanting, see the tutorials for more information on choosing one of these

- keep_p_gen_dist (bool): if possible, whether to keep the distribution of generation probabilities of the sequences the same as provided by the model without any signals

- p_gen_bin_count (int): if keep_p_gen_dist is true, how many bins to use to approximate the generation probability distribution

- remove_seqs_with_signals (bool): if true, it explicitly controls the proportions of signals in sequences and removes any accidental occurrences

- species (str): species that the sequences come from; used to select correct genes to export full length sequences; default is 'human'

- implanting_scaling_factor (int): determines in how many receptors to implant the signal in reach iteration; this is computed as number_of_receptors_needed_for_signal * implanting_scaling_factor; useful when using Implanting simulation strategy in combination with importance sampling, since the generation probability of some receptors with implanted signals might be very rare and those receptors might end up not being kept often with importance sampling; this parameter is only used when keep_p_gen_dist is set to True


**YAML specification:**

.. indent-with-spaces
.. code-block:: yaml

    definitions:
        simulations:
            sim1:
                is_repertoire: false
                paired: false
                sequence_type: amino_acid
                simulation_strategy: RejectionSampling
                sim_items:
                    sim_item1: # group of sequences with same simulation params
                        generative_model:
                            chain: beta
                            default_model_name: humanTRB
                            model_path: null
                            type: OLGA
                        number_of_examples: 100
                        seed: 1002
                        signals:
                            signal1: 1



Simulation config item
''''''''''''''''''''''''''''''''''''''''''''''''''''


When performing a simulation, one or more simulation config items can be specified.
Config items define groups of repertoires or receptors that have the same simulation parameters,
such as signals, generative model, clonal frequencies, noise parameters.


**Specification arguments:**

- signals (dict): signals for the simulation item and the proportion of sequences in the repertoire that will have the given signal. For receptor-level simulation, the proportion will always be 1.

- is_noise (bool): indicates whether the implanting should be regarded as noise; if it is True, the signals will be implanted as specified, but the repertoire/receptor in question will have negative class.

- generative_model: parameters of the generative model, including its type, path to the model; currently supported models are OLGA and ExperimentalImport

- seed (int): starting random seed for the generative model (it should differ across simulation items, or it can be set to null when not used)

- false_positives_prob_in_receptors (float): when performing repertoire level simulation, what percentage of sequences should be false positives

- false_negative_prob_in_receptors (float): when performing repertoire level simulation, what percentage of sequences should be false negatives

- immune_events (dict): a set of key-value pairs that will be added to the metadata (same values for all data generated in one simulation sim_item) and can be later used as labels

- default_clonal_frequency (dict): clonal frequency in Ligo is simulated through `scipy's zeta distribution function for generating random numbers <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zipf.html>`_, with parameters provided under default_clonal_frequency parameter. These parameters will be used to assign count values to sequences that do not contain any signals if they are required by the simulation. If clonal frequency shouldn't be used, this parameter can be None

.. indent with spaces
.. code-block:: yaml

    clonal_frequency:
        a: 2 # shape parameter of the distribution
        loc: 0 # 0 by default but can be used to shift the distribution

- sequence_len_limits (dict): allows for filtering the generated sequences by length, needs to have parameters min and max specified; if not used, min/max should be -1

.. indent with spaces
.. code-block:: yaml

    sequence_len_limits:
        min: 4 # keep sequences of length 4 and longer
        max: -1 # no limit on the max length of the sequences

**YAML specification:**

.. indent with spaces
.. code-block:: yaml

    definitions:
        simulations: # definitions of simulations should be under key simulations in the definitions part of the specification
            # one simulation with multiple implanting objects, a part of definition section
            my_simulation:
                sim_item1:
                    number_of_examples: 10
                    seed: null # don't use seed
                    receptors_in_repertoire_count: 100
                    generative_model:
                        chain: beta
                        default_model_name: humanTRB
                        model_path: null
                        type: OLGA
                    signals:
                        my_signal: 0.25
                        my_signal2: 0.01
                        my_signal__my_signal2: 0.02 # my_signal and my_signal2 will co-occur in 2% of the receptors in all 10 repertoires
                sim_item2:
                    number_of_examples: 5
                    receptors_in_repertoire_count: 150
                    seed: 10 #
                    generative_model:
                        chain: beta
                        default_model_name: humanTRB
                        model_path: null
                        type: OLGA
                    signals:
                        my_signal: 0.75
                    default_clonal_frequency:
                        a: 2
                    sequence_len_limits:
                        min: 3



