class Constants:

    VERSION = "3.0.27"

    # encoding constants
    FEATURE_DELIMITER = "_"

    # importing constants
    UNKNOWN = ''
    ALLELE_DELIMITER = "*"
    GENE_DELIMITER = "-"
    STOP_CODON = "*"
    CACHE_TYPE = "cache_type"
    CACHE_PATH = 'cache_path'
    # env variable: if set to "true", HTMLBuilder pickles instruction states before building the HTML (for debugging)
    STORE_INSTRUCTION_STATES = 'store_instruction_states' # TODO: in future release, either make immuneML runs possible to continue or drop this entirely
    COMMENT_SIGN = "#"
    NOT_COMPUTED = "not computed"
    SIGNAL_DELIMITER = "__"
    GAP_LETTER = "-"

    # plotting constants
    PLOTLY_BLACK = "#2A3F5E"


VERSION = Constants.VERSION
