from source.dsl.ObjectParser import ObjectParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.logging.Logger import log
from source.simulation.implants.Motif import Motif
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class MotifParser:

    @staticmethod
    def parse_motifs(motifs: dict, symbol_table: SymbolTable):

        valid_motif_keys = ["seed", "instantiation"]
        for key in motifs.keys():

            ParameterValidator.assert_keys(motifs[key].keys(), valid_motif_keys, "MotifParser", key)

            motif = MotifParser._parse_motif(key, motifs[key])
            symbol_table.add(key, SymbolType.MOTIF, motif)

        return symbol_table, motifs

    @staticmethod
    @log
    def _parse_motif(key: str, motif_item: dict) -> Motif:

        classes = ReflectionHandler.get_classes_by_partial_name("Instantiation", "motif_instantiation_strategy/")
        valid_values = [cls.__name__[:-13] for cls in MotifInstantiationStrategy.__subclasses__()]
        instantiation_object = ObjectParser.parse_object(motif_item["instantiation"], valid_values, "Instantiation",
                                                         "motif_instantiation_strategy", "MotifParser", key)
        motif = Motif(key, instantiation_object, seed=motif_item["seed"])

        return motif
