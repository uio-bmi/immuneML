import importlib

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


class ClusteringInstruction(Instruction):
    """
    Todo: add documentation here
    """

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
        encoded_dataset.name = unit.dataset.name
        if unit.dim_red_before_clustering:
            self._dim_reduce(unit, encoded_dataset)

        unit.clustering_method.fit(encoded_dataset.encoded_data)

        if not unit.dim_red_before_clustering:
            self._dim_reduce(unit, encoded_dataset)

        if unit.eval_metrics is not None:
            labels_true = None
            if unit.true_labels_path is not None and unit.true_labels_path.is_file():
                try:
                    labels_true = genfromtxt(unit.true_labels_path, dtype=int, delimiter=',')
                except:
                    print_log("Problem getting true_labels_path file\nCheck the file is in the right format(CSV, 1 line)")
            distance_metric = unit.clustering_method.model.metric if hasattr(unit.clustering_method.model, "metric") else "euclidean"
            self.calculate_scores(key, encoded_dataset.encoded_data.examples, unit.clustering_method.model.labels_, labels_true, unit.eval_metrics, distance_metric)

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
        encoded_dataset = DataEncoder.run(
            DataEncoderParams(
                dataset=unit.dataset,
                encoder=unit.encoder,
                encoder_params=EncoderParams(result_path=result_path,
                                             label_config=unit.label_config,
                                             pool_size=unit.number_of_processes,
                                             filename="encoded_dataset.pkl",
                                             learn_model=True,
                                             encode_labels=unit.label_config is not None
                                             )
            )
        )
        return encoded_dataset

    def _dim_reduce(self, unit: ClusteringUnit, dataset):
        if unit.dimensionality_reduction is not None:
            unit.dimensionality_reduction.fit_transform(dataset.encoded_data)

    def calculate_scores(self, key, data, labels_pred, labels_true, metrics, distance_metric):
        if self.state.clustering_scores is None:
            self.state.clustering_scores = {"target_score": {}}

        if isinstance(data, csr_matrix):
            data = data.toarray()

        scores = {}

        # Check if more than one cluster
        if max(labels_pred) <= 0:
            for metric in metrics:
                scores[metric] = "Only 1 cluster"
        else:
            from sklearn.metrics import (
                silhouette_score,
                calinski_harabasz_score,
                davies_bouldin_score,
                adjusted_rand_score,
                adjusted_mutual_info_score,
                homogeneity_score,
                completeness_score,
                v_measure_score,
                fowlkes_mallows_score
            )

            metric_functions = {
                "Silhouette": silhouette_score,
                "Calinski-Harabasz": calinski_harabasz_score,
                "Davies-Bouldin": davies_bouldin_score,
                "Rand index": adjusted_rand_score,
                "Mutual Information": adjusted_mutual_info_score,
                "Homogeneity": homogeneity_score,
                "Completeness": completeness_score,
                "V-measure": v_measure_score,
                "Fowlkes-Mallows": fowlkes_mallows_score
            }

            for metric in metrics:
                if metric in metric_functions:
                    if labels_true is not None and metric in ["Rand index", "Mutual Information", "Homogeneity", "Completeness", "V-measure", "Fowlkes-Mallows"]:
                        scores[metric] = metric_functions[metric](labels_true, labels_pred)
                    elif metric in ["Silhouette"]:
                        scores[metric] = metric_functions[metric](data, labels_pred, metric=distance_metric)
                    else:
                        scores[metric] = metric_functions[metric](data, labels_pred)

        target_scores = {
            "Silhouette": 1,
            "Calinski-Harabasz": 0,
            "Davies-Bouldin": 999999,
            "Rand index": 1,
            "Mutual Information": 1,
            "Homogeneity": 1,
            "Completeness": 1,
            "V-measure": 1,
            "Fowlkes-Mallows": 1
        }
        self.state.clustering_scores["target_score"].update({
            metric: target_scores[metric] for metric in metrics
        })

        self.state.clustering_scores[key] = scores

    def add_label(self, dataset: Dataset, labels: [str], path: Path):
        print_log("Started copying dataset...", include_datetime=True)
        PathBuilder.build(path)

        if type(dataset).__name__ == "RepertoireDataset":
            processed_dataset = self._process_repertoire_dataset(dataset, labels, path)
        elif type(dataset).__name__ == "ReceptorDataset":
            processed_dataset = self._process_receptor_dataset(dataset, labels, path)
        else:
            processed_dataset = self._process_sequence_dataset(dataset, labels, path)

        processed_dataset.encoded_data.labels["cluster_id"] = labels
        processed_dataset.labels["cluster_id"] = list(range(max(labels) + 1))
        print_log("Finished copying dataset.", include_datetime=True)
        return processed_dataset

    def _process_repertoire_dataset(self, dataset, labels, path):
        from immuneML.data_model.repertoire.Repertoire import Repertoire
        from immuneML.util.ImportHelper import ImportHelper
        from yaml import dump

        repertoires_path = path / "repertoires"
        PathBuilder.build(repertoires_path)
        dataset_name = dataset.name

        repertoires = []
        for index, x in enumerate(dataset.get_data()):
            filename = x.data_filename.stem
            new_repertoire = Repertoire.build_like(repertoire=x, indices_to_keep=list(range(x.get_element_count())), result_path=repertoires_path, filename_base=filename)
            new_repertoire.metadata["cluster_id"] = str(labels[index])
            repertoires.append(new_repertoire)

            metadata_filename = repertoires_path / f"{filename}_metadata.yaml"
            with metadata_filename.open("w") as file:
                dump(new_repertoire.metadata, file)

        df = dataset.get_metadata(dataset.get_metadata_fields(), return_df=True)
        df.insert(0, "cluster_id", labels.tolist(), True)

        metadata_filename = ImportHelper.make_new_metadata_file(repertoires=repertoires, metadata=df, result_path=path, dataset_name=dataset_name)

        processed_dataset = dataset
        processed_dataset.repertoires = repertoires
        processed_dataset.metadata_file = metadata_filename
        processed_dataset.metadata_fields.append("cluster_id")
        return processed_dataset

    def _process_receptor_dataset(self, dataset, labels, path):
        from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset

        processed_receptors = [x for x in dataset.get_data()]
        for index, receptor in enumerate(processed_receptors):
            for seq in receptor.get_chains():
                receptor.__dict__[seq].metadata.custom_params["cluster_id"] = str(labels[index])
            receptor.metadata["cluster_id"] = str(labels[index])

        processed_dataset = ReceptorDataset.build_from_objects(receptors=processed_receptors, file_size=dataset.file_size, name=dataset.name, path=path)
        processed_dataset.encoded_data = dataset.encoded_data
        processed_dataset.labels = dataset.labels
        return processed_dataset

    def _process_sequence_dataset(self, dataset, labels, path):
        from immuneML.data_model.dataset.SequenceDataset import SequenceDataset

        processed_sequences = [x for x in dataset.get_data()]
        for index, seq in enumerate(processed_sequences):
            seq.metadata.custom_params["cluster_id"] = str(labels[index])

        processed_dataset = SequenceDataset.build_from_objects(sequences=processed_sequences, file_size=dataset.file_size, name=dataset.name, path=path)
        processed_dataset.encoded_data = dataset.encoded_data
        processed_dataset.labels = dataset.labels
        return processed_dataset
