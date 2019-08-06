from source.dsl.SymbolTable import SymbolTable


class MatchingSequenceDetailsParser:

    @staticmethod
    def parse(params: dict, symbol_table: SymbolTable):
        return {
            "reference_sequences": symbol_table.get_config(params["encoding"])["reference_sequences"],
            "max_distance": symbol_table.get_config(params["encoding"])["max_distance"]
        }, {"encoding": params["encoding"]}
