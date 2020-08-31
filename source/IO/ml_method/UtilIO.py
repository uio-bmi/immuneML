import pickle
import shutil

from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.pairwise_repertoire_comparison.PairwiseRepertoireComparison import PairwiseRepertoireComparison


class UtilIO:

    @staticmethod
    def export_comparison_data(comp_data: ComparisonData, path: str):
        shutil.copytree(comp_data.path, f"{path}comp_data/")

        comp_data_file = f"{path}comparison_data.pickle"
        with open(comp_data_file, 'wb') as file:
            pickle.dump(comp_data, file)

        return comp_data_file

    @staticmethod
    def import_comparison_data(path: str) -> ComparisonData:
        comp_data_path = f"{path}comp_data/"

        comp_data_file = f"{path}comparison_data.pickle"
        with open(comp_data_file, "rb") as file:
            comp_data = pickle.load(file)

        comp_data.path = comp_data_path
        for batch in comp_data.batches:
            batch.path = comp_data_path

        return comp_data

    @staticmethod
    def export_pairwise_comparison(pairwise_comparison: PairwiseRepertoireComparison, path: str) -> str:
        pairwise_comparison_file = f"{path}pairwise_repertoire_comparison.pickle"
        with open(pairwise_comparison_file, "wb") as file:
            pickle.dump(pairwise_comparison, file)

        UtilIO.export_comparison_data(pairwise_comparison.comparison_data, path)

        return pairwise_comparison_file

    @staticmethod
    def import_pairwise_comparison(path: str) -> PairwiseRepertoireComparison:
        filename = f"{path}pairwise_repertoire_comparison.pickle"
        with open(filename, "rb") as file:
            pairwise_comparison = pickle.load(file)

        pairwise_comparison.comparison_data = UtilIO.import_comparison_data(path)
        pairwise_comparison.path = path

        return pairwise_comparison
