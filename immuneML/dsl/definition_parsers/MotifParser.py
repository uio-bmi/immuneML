import copy

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.dsl.ObjectParser import ObjectParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class MotifParser:

    @staticmethod
    def parse_motifs(motifs: dict, symbol_table: SymbolTable):

        valid_motif_keys = ["seed", "instantiation", "seed_chain1", "seed_chain2", "name_chain1", "name_chain2"]
        for key in motifs.keys():

            ParameterValidator.assert_keys(motifs[key].keys(), valid_motif_keys, "MotifParser", key, exclusive=False)

            motif = MotifParser._parse_motif(key, motifs[key])
            symbol_table.add(key, SymbolType.MOTIF, motif)

        return symbol_table, motifs

    @staticmethod
    @log
    def _parse_motif(key: str, motif_item: dict) -> Motif:

        motif_dict = copy.deepcopy(motif_item)

        valid_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MotifInstantiationStrategy, "Instantiation", "motif_instantiation_strategy/")
        instantiation_object = ObjectParser.parse_object(motif_item["instantiation"], valid_values, "Instantiation",
                                                         "motif_instantiation_strategy", "MotifParser", key)
        motif_dict["instantiation"] = instantiation_object
        motif_dict["identifier"] = key

        if "name_chain1" in motif_item:
            motif_dict["name_chain1"] = Chain[motif_item["name_chain1"].upper()]
        if "name_chain2" in motif_item:
            motif_dict["name_chain2"] = Chain[motif_item["name_chain2"].upper()]

        assert "seed" in motif_dict or all(el in motif_dict for el in ["name_chain1", "name_chain2", "seed_chain1", "seed_chain2"]), \
            "MotifParser: please check the documentation for motif definition. Either parameter `seed` has to be set (for simulation in single " \
            "chain data) or all of the parameters `name_chain1`, `name_chain2`, `seed_chain1`, `seed_chain2` (for simulation for paired chain data)."

        motif = Motif(**motif_dict)

        return motif
