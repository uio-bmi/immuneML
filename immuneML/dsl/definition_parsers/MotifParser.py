import copy

from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.implants.LigoPWM import LigoPWM
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class MotifParser:

    keyword = "motifs"

    @staticmethod
    def parse(motifs: dict, symbol_table: SymbolTable):

        for key, motif_dict in motifs.items():

            motif_keys = list(motif_dict.keys())
            if "seed" in motif_keys:
                ParameterValidator.assert_keys(motif_keys, ['seed', 'instantiation'], "MotifParser", key, exclusive=False)
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

        valid_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MotifInstantiationStrategy, "Instantiation", "motif_instantiation_strategy/")
        instantiation_object = ObjectParser.parse_object(motif_item["instantiation"], valid_values, "Instantiation",
                                                         "motif_instantiation_strategy", "MotifParser", key)
        motif_dict["instantiation"] = instantiation_object
        motif_dict["identifier"] = key

        assert "seed" in motif_dict or all(el in motif_dict for el in ["name_chain1", "name_chain2", "seed_chain1", "seed_chain2"]), \
            "MotifParser: please check the documentation for motif definition. Either parameter `seed` has to be set (for simulation in single " \
            "chain data) or all of the parameters `name_chain1`, `name_chain2`, `seed_chain1`, `seed_chain2` (for simulation for paired chain data)."

        motif = SeedMotif(**motif_dict)

        return motif
