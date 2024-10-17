import pickle
import shutil
from pathlib import Path

from immuneML.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison


class UtilIO:

    @staticmethod
    def export_pairwise_comparison(pairwise_comparison: PairwiseRepertoireComparison, path: Path) -> Path:
        pairwise_comparison_file = path / "pairwise_repertoire_comparison.pickle"
        with pairwise_comparison_file.open("wb") as file:
            pickle.dump(pairwise_comparison, file)

        UtilIO.export_comparison_data(pairwise_comparison.comparison_data, path)

        return pairwise_comparison_file

    @staticmethod
    def import_pairwise_comparison(path: Path) -> PairwiseRepertoireComparison:
        filename = path / "pairwise_repertoire_comparison.pickle"
        with filename.open("rb") as file:
            pairwise_comparison = pickle.load(file)

        pairwise_comparison.comparison_data = UtilIO.import_comparison_data(path)
        pairwise_comparison.path = path

        return pairwise_comparison
