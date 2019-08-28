import abc


class Receptor:

    @abc.abstractmethod
    def get_chains(self):
        pass

    def get_chain(self, chain: str):
        return getattr(self, chain)
