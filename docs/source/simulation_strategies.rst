######################
Simulation strategies
######################

.. toctree::
   :maxdepth: 2

Signal Implanting
=================

Signal implanting strategies define which sequences will be chosen from a repertoire to implant the signal in and
once the sequences are chosen, they define where in the sequence the specific motif instance will be introduced.

Signal implanting strategy package consist of an abstract class **SignalImplantingStrategy** defining the methods
each individual implanting strategy has to offer.

Implanting only in healthy sequences
====================================

The **HealthySequenceImplanting** class implements **SignalImplantingStrategy**. HealthySequenceImplanting chooses only
sequences which do not have any other signal for signal implanting. In addition to not having any other signals,
sequences chosen for implanting have to be longer than the longest possible motif instance for the signal. In this way,
when a motif instance is implanted into a sequence, the sequence retains its original length.

When creating an instance of *HealthySequenceImplanting* class, the following parameters should be defined:

*   sequence implanting and
*   sequence position weights.

The sequence position weights defines for each position in a sequence how likely it is that the concrete motif instance
will be implanted starting at that position. The positions are defined according to the IMGT [1]_. An example where the
motif instance is most likely to be at the first position and less likely to be at the second and fifth position
in a CDR3 sequence is:

.. code-block:: python

    sequence_position_weights = {
        105: 0.8,
        106: 0.1,
        107: 0,
        108: 0,
        109: 0.1,
        113: 0,
        114: 0,
        115: 0,
        116: 0,
        117: 0
    }

In case that CDR3 sequence chosen for implanting is only 6 amino acids long (and therefore includes only positions
105, 106, 107, 115, 116 and 117), the probabilities of implanting from ``sequence_position_weights`` will be kept only for
those positions that exist and normalized in order to represent probabilities.

However, *HealthySequenceImplanting* defines only how to choose sequences for implanting, but not where the in the sequence
to implant. For that, *SequenceImplantingStrategy* is used.

Sequence Implanting
===================

The **GappedMotifImplanting** class inherits *SequenceImplantingStrategy* and implements its abstract method for
implanting a motif instance in sequence. This class allows for implanting both motif instances with gaps and without
gaps (by setting gap parameter to be equal to 0). It implements the following function:

.. code-block:: python

    def implant(self, sequence: Sequence, signal: dict, sequence_position_weights=None) -> Sequence

It accepts a *Sequence* object in which the motif instance will be implanted, ``sequence_position_weights`` as defined
in :ref:`Implanting only in healthy sequences` section. If no sequence_position_weights is specified, then all positions
in the sequence are equally likely to be the starting position for the motif instance implanting.
The second parameter, signal dictionary, has the following format:

.. code-block:: python

    {
        "signal_id": "signal1",                                 # a unique signal identifier
        "motif_id": "motif1",                                   # a unique motif identifier
        "motif_instance": MotifInstance(instance="CAS", gap=0)  # an object of MotifInstance class
    }

The signal dictionary is used to supply information to the new sequence about the specific signal, motif and motif
instance that are implanted in the original sequence. This information will be stored in an object of the **Implant** class
in the ``annotation`` attribute of the *Sequence* object. The annotation is stored as the *SequenceAnnotation* instance
which contain a list of implants.

Implant class
=============

The **Implant** class encapsulates information about the implanted signal. It consists of:

*   a unique signal identifier (signal_id),
*   a unique motif identifier (motif_id),
*   a MotifInstance object and
*   a position.

The position parameter stores the probabilistically chosen position where the motif instance was implanted.

Motif Instantiation
===================

Previous sections describe how the motif instance will be implanted in the sequence, whereas this section focuses on
the way a motif instance will be created.

To allow for greater flexibility, a disease-specific signal consists of a list of different motifs each of which is
defined in a probabilistic manner. For more information on the simulation data model, see :ref:`Simulation model`.

When performing motif instantiation, a motif is randomly chosen from the signal. The probability of choosing a motif is
uniform across motifs.

Once a motif is chosen, a specific instance is built using a motif instantiation strategy.

Motif Instantiation Strategy
============================

Motif instantiation strategy is implemented in one of the two ways:

*   a motif instance is always the same as the motif seed: **IdentityMotifInstantiation** or
*   a motif instance is created according to the given specifications: **GappedKmerInstantiation**.

*GappedKmerInstantiation* allows for the definition of the parameters:

*   max_hamming_distance: maximum number of letters in which the motif instance can differ from the motif seed,
*   max_gap: if a motif seed allows for the gap, this is the maximum size of the gap,
*   min_gap: if a motif seed allows for the gap, this is the minimum size of the gap.

The gap size for each motif instance is chosen randomly from uniform distribution within the specified limits.

The Hamming distance of the motif instance from the original seed is also chosen from a uniform distribution between
0 and ``max_hamming_distance``. The letter (e.g. amino acids) which will substitute the ones from the seed are chosen
in accordance with the probability specified on motif creation in the parameter ``alphabet_weights``. For more on this
parameter, see :ref:`Motif`.

.. [1] Lefranc, M.-P., The Immunologist, 7, 132-136 (1999).