import inspect

from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
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
    def _parse_instantiation(key: str, motif_item: dict):
        classes = ReflectionHandler.get_classes_by_partial_name("Instantiation", "motif_instantiation_strategy/")
        valid_values = [cls.__name__[:-13] for cls in MotifInstantiationStrategy.__subclasses__()]

        assert len(motif_item["instantiation"]) == 1, \
            f"MotifParser: More than one parameter passed to instantiation for motif under key {key}. " \
            f"Only one value can be specified here. Valid options are: {str(valid_values)[1:-1]}"

        if isinstance(motif_item["instantiation"], set):
            params = {}
            instantiation_class_name = list(motif_item["instantiation"])[0]
        else:
            instantiation_class_name = list(motif_item["instantiation"].keys())[0]
            params = motif_item["instantiation"][instantiation_class_name]

        ParameterValidator.assert_in_valid_list(instantiation_class_name, valid_values, "MotifParser", "instantiation")
        instantiation_class = ReflectionHandler.get_class_by_name("{}Instantiation".format(instantiation_class_name))

        return instantiation_class, params

    @staticmethod
    @log
    def _parse_motif(key: str, motif_item: dict) -> Motif:
        instantiation_class, params = MotifParser._parse_instantiation(key, motif_item)

        try:
            instantiation_object = instantiation_class(**params)
            motif = Motif(key, instantiation_object,
                          seed=motif_item["seed"])
        except TypeError as err:
            print(f"MotifParser: invalid parameter {err.args[0]} when specifying parameters in {motif_item['instantiation']} "
                  f"under motif {key}. Valid parameter names are: "
                  f"{[name for name in inspect.signature(instantiation_class.__init__).keys()]}")
            raise err

        return motif
