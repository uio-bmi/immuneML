import copy

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.implants.LigoPWM import LigoPWM
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator


class MotifParser:

    keyword = "motifs"

    @staticmethod
    def parse(motifs: dict, symbol_table: SymbolTable):

        for key, motif_dict in motifs.items():

            motif_keys = list(motif_dict.keys())
            if "seed" in motif_keys:
                ParameterValidator.assert_keys(motif_keys, ['seed', 'min_gap', 'max_gap', 'hamming_distance_probabilities', 'position_weights', 'alphabet_weights'], "MotifParser", key, exclusive=False)
                motif = MotifParser._parse_seed_motif(key, motif_dict)
            else:
                ParameterValidator.assert_keys(motif_keys, ['file_path', 'threshold'], "MotifParser", key)
                motif = LigoPWM.build(identifier=key, file_path=motif_dict['file_path'], threshold=motif_dict['threshold'])

            symbol_table.add(key, SymbolType.MOTIF, motif)

        return symbol_table, motifs

    @staticmethod
    @log
    def _parse_seed_motif(key: str, motif_item: dict) -> Motif:

        motif_dict = copy.deepcopy(motif_item)
        motif_dict["identifier"] = key

        if 'hamming_distance_probabilities' in motif_dict and motif_dict['hamming_distance_probabilities'] is not None:
            motif_dict['hamming_distance_probabilities'] = {key: float(value) for key, value in motif_dict['hamming_distance_probabilities'].items()}
            assert all(isinstance(key, int) for key in motif_dict['hamming_distance_probabilities'].keys()) \
                   and all(isinstance(val, float) for val in motif_dict['hamming_distance_probabilities'].values()) \
                   and 0.99 <= sum(motif_dict['hamming_distance_probabilities'].values()) <= 1, \
                   "For each possible Hamming distance a probability between 0 and 1 has to be assigned " \
                   "so that the probabilities for all distance possibilities sum to 1."

        motif = SeedMotif(**motif_dict)

        return motif
