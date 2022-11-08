from pathlib import Path
from numpy import genfromtxt

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState
from immuneML.workflows.instructions.clustering.ClusteringUnit import ClusteringUnit
from immuneML.workflows.steps.DataEncoder import DataEncoder
from immuneML.workflows.steps.DataEncoderParams import DataEncoderParams
from immuneML.util.Logger import print_log

from scipy.sparse import csr_matrix

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, \
    fowlkes_mallows_score


class ClusteringInstruction(Instruction):
    def __init__(self, clustering_units: dict, name: str = None):
        assert all(isinstance(unit, ClusteringUnit) for unit in clustering_units.values()), \
            "ClusteringInstruction: not all elements passed to init method are instances of ClusteringUnit."
        self.state = ClusteringState(clustering_units, name=name)
        self.name = name

    def run(self, result_path: Path):
        name = self.name if self.name is not None else "clustering"
        self.state.result_path = result_path / name
        for index, (key, unit) in enumerate(self.state.clustering_units.items()):
            print_log("Started analysis {} ({}/{}).".format(key, index + 1, len(self.state.clustering_units)), include_datetime=True)
            path = self.state.result_path / f"analysis_{key}"
            PathBuilder.build(path)
            report_result = self.run_unit(unit, path, key)
            unit.report_result = report_result
            print_log("Finished analysis {} ({}/{}).\n".format(key, index + 1, len(self.state.clustering_units)), include_datetime=True)
        return self.state

    def run_unit(self, unit: ClusteringUnit, result_path: Path, key) -> ReportResult:
        encoded_dataset = self.encode(unit, result_path / "encoded_dataset")

        if unit.dim_red_before_clustering:
            if unit.dimensionality_reduction is not None:
                unit.dimensionality_reduction.fit(encoded_dataset.encoded_data)
                unit.dimensionality_reduction.transform(encoded_dataset.encoded_data)

        unit.clustering_method.fit(encoded_dataset.encoded_data)

        # Check if more than 1 cluster
        if max(unit.clustering_method.model.labels_) > 0:
            labels_true = None
            if unit.true_labels_path is not None:
                if unit.true_labels_path.is_file():
                    try:
                        labels_true = genfromtxt(unit.true_labels_path, dtype=int, delimiter=',')
                    except:
                        print_log("Problem getting true_labels_path file\nCheck the file is in the right format(CSV in 1 line)")
            self.calculate_scores(key, encoded_dataset.encoded_data.examples, unit.clustering_method.model.labels_, labels_true)

        if not unit.dim_red_before_clustering:
            if unit.dimensionality_reduction is not None:
                unit.dimensionality_reduction.fit(encoded_dataset.encoded_data)
                unit.dimensionality_reduction.transform(encoded_dataset.encoded_data)

        processed_dataset = self.add_label(encoded_dataset, unit.clustering_method.model.labels_, result_path / "dataset_clustered")

        unit.report.dataset = processed_dataset
        unit.report.method = unit.clustering_method
        unit.report.result_path = result_path / "report"
        unit.report.number_of_processes = unit.number_of_processes
        print_log("Report generation started...", include_datetime=True)
        report_result = unit.report.generate_report()
        print_log("Report generating finished.", include_datetime=True)
        return report_result

    def encode(self, unit: ClusteringUnit, result_path: Path) -> Dataset:
        if unit.encoder is not None:
            encoded_dataset = DataEncoder.run(DataEncoderParams(dataset=unit.dataset, encoder=unit.encoder,
                                                                encoder_params=EncoderParams(result_path=result_path,
                                                                                             label_config=unit.label_config,
                                                                                             pool_size=unit.number_of_processes,
                                                                                             filename="encoded_dataset.pkl",
                                                                                             learn_model=True,
                                                                                             encode_labels=unit.label_config is not None),
                                                                ))
        else:
            encoded_dataset = unit.dataset
        return encoded_dataset

    def calculate_scores(self, key, data, labels_pred, labels_true):
        if self.state.clustering_scores is None:
            self.state.clustering_scores = {}

        if isinstance(data, csr_matrix):
            data = data.toarray()

        scores = {
            "Silhouette": silhouette_score(data, labels_pred),
            "Calinski-Harabasz": calinski_harabasz_score(data, labels_pred),
            "Davies-Bouldin": davies_bouldin_score(data, labels_pred)
        }
        if "target_score" not in self.state.clustering_scores.keys():
            self.state.clustering_scores["target_score"] = {
                "Silhouette": 1,
                "Calinski-Harabasz": 999999,
                "Davies-Bouldin": 0
            }

        if labels_true is not None:
            labels_true_scores = {
                "Rand index": adjusted_rand_score(labels_true, labels_pred),
                "Mutual Information": adjusted_mutual_info_score(labels_true, labels_pred),
                "Homogeneity": homogeneity_score(labels_true, labels_pred),
                "Completeness": completeness_score(labels_true, labels_pred),
                "V-measure": v_measure_score(labels_true, labels_pred),
                "Fowlkes-Mallows": fowlkes_mallows_score(labels_true, labels_pred)
            }
            if "Rand index" not in self.state.clustering_scores["target_score"]:
                labels_true_target_score = {
                    "Rand index": 1,
                    "Mutual Information": 1,
                    "Homogeneity": 1,
                    "Completeness": 1,
                    "V-measure": 1,
                    "Fowlkes-Mallows": 1
                }
                self.state.clustering_scores["target_score"].update(labels_true_target_score)
            scores.update(labels_true_scores)

        self.state.clustering_scores[key] = scores

    def add_label(self, dataset: Dataset, labels: [str], path: Path):
        PathBuilder.build(path)
        if type(dataset).__name__ == "RepertoireDataset":
            # TO DO: Fix building for repertoire dataset
            for index, x in enumerate(dataset.get_data()):
                x.metadata["cluster_id"] = str(labels[index])

            processed_dataset = dataset
        else:
            processed_receptors = [x for x in dataset.get_data()]
            for index, receptor in enumerate(processed_receptors):
                for seq in receptor.get_chains():
                    receptor.__dict__[seq].metadata.custom_params["cluster_id"] = str(labels[index])
                receptor.metadata["cluster_id"] = str(labels[index])

            processed_dataset = ReceptorDataset.build_from_objects(receptors=processed_receptors, file_size=dataset.file_size, name=dataset.name, path=path)
            processed_dataset.labels = dataset.labels
            processed_dataset.encoded_data = dataset.encoded_data
            processed_dataset.encoded_data.labels["cluster_id"] = labels

        processed_dataset.labels["cluster_id"] = list(range(max(labels)+1))
        return processed_dataset
