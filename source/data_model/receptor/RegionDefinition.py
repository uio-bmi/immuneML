from enum import Enum


class RegionDefinition(Enum):
    """
    Class describing different definitions for regions:
    e.g. IMGT and AIRR define CDR3 excluding the conserved cysteine and tryptophan/phenylalanine residues, while JUNCTION includes those.

    For more details on definitions for AIRR standards, see:
    https://docs.airr-community.org/en/stable/datarep/rearrangements.html#definition-clarifications
    """

    IMGT = "IMGT"
