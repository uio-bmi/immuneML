from source.dsl.SymbolTable import SymbolTable


class MatchingSequenceDetailsParser:

    @staticmethod
    def parse(params: dict, symbol_table: SymbolTable) -> dict:
        return {
            "encoding": params["encoding"],
            "reference_sequences": symbol_table.get(params["encoding"])["params"]["reference_sequences"],
            "max_distance": symbol_table.get(params["encoding"])["params"]["max_distance"]
        }