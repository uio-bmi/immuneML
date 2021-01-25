import pickle
import shutil
from pathlib import Path

from immuneML.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from immuneML.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison


class UtilIO:

    @staticmethod
    def export_comparison_data(comp_data: ComparisonData, path: Path):
        shutil.copytree(comp_data.path, path / "comp_data")

        comp_data_file = path / "comparison_data.pickle"
        with comp_data_file.open('wb') as file:
            pickle.dump(comp_data, file)

        return comp_data_file

    @staticmethod
    def import_comparison_data(path: Path) -> ComparisonData:
        comp_data_path = path / "comp_data"

        comp_data_file = path / "comparison_data.pickle"
        with comp_data_file.open("rb") as file:
            comp_data = pickle.load(file)

        comp_data.path = comp_data_path
        for batch in comp_data.batches:
            batch.path = comp_data_path

        return comp_data

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
