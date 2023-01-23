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

        labels_true = None
        if unit.true_labels_path is not None and unit.true_labels_path.is_file():
            try:
                labels_true = genfromtxt(unit.true_labels_path, dtype=int, delimiter=',')
            except:
                print_log("Problem getting true_labels_path file\nCheck the file is in the right format(CSV, 1 line)")
        self.calculate_scores(key, encoded_dataset.encoded_data.examples, unit.clustering_method.model.labels_, labels_true)

        if not unit.dim_red_before_clustering:
            self._dim_reduce(unit, encoded_dataset)

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

    def _dim_reduce(self, unit: ClusteringUnit, dataset):
        if unit.dimensionality_reduction is not None:
            unit.dimensionality_reduction.fit(dataset.encoded_data)
            unit.dimensionality_reduction.transform(dataset.encoded_data)

    def calculate_scores(self, key, data, labels_pred, labels_true):
        if self.state.clustering_scores is None:
            self.state.clustering_scores = {}

        if isinstance(data, csr_matrix):
            data = data.toarray()

        #Check if more than one cluster
        if max(labels_pred) <= 0:
            scores = {
                "Silhouette": "Only 1 cluster",
                "Calinski-Harabasz": "Only 1 cluster",
                "Davies-Bouldin": "Only 1 cluster"
            }
        else:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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
            if max(labels_pred) <= 0:
                labels_true_scores = {
                    "Rand index": "Only 1 cluster",
                    "Mutual Information": "Only 1 cluster",
                    "Homogeneity": "Only 1 cluster",
                    "Completeness": "Only 1 cluster",
                    "V-measure": "Only 1 cluster",
                    "Fowlkes-Mallows": "Only 1 cluster"
                }
            else:
                from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
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
        print_log("Started copying dataset...", include_datetime=True)
        PathBuilder.build(path)
        if type(dataset).__name__ == "RepertoireDataset":
            from immuneML.data_model.repertoire.Repertoire import Repertoire
            from immuneML.util.ImportHelper import ImportHelper
            from yaml import dump
            repertoiresPath = path / "repertoires"
            PathBuilder.build(repertoiresPath)
            dataset_name = dataset.name

            repertoires = []
            for index, x in enumerate(dataset.get_data()):
                filename = x.data_filename.stem
                newRepertoire = Repertoire.build_like(repertoire=x, indices_to_keep=list(range(x.get_element_count())), result_path=repertoiresPath, filename_base=filename)
                newRepertoire.metadata["cluster_id"] = str(labels[index])
                repertoires.append(newRepertoire)

                metadata_filename = repertoiresPath / f"{filename}_metadata.yaml"
                with metadata_filename.open("w") as file:
                    dump(newRepertoire.metadata, file)

            df = dataset.get_metadata(dataset.get_metadata_fields(), return_df=True)
            df.insert(0, "cluster_id", labels.tolist(), True)

            metadata_filename = ImportHelper.make_new_metadata_file(repertoires=repertoires, metadata=df, result_path=path, dataset_name=dataset_name)

            processed_dataset = dataset
            processed_dataset.repertoires = repertoires
            processed_dataset.metadata_file = metadata_filename
            processed_dataset.metadata_fields.append("cluster_id")
        elif type(dataset).__name__ == "ReceptorDataset":
            from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
            processed_receptors = [x for x in dataset.get_data()]
            for index, receptor in enumerate(processed_receptors):
                for seq in receptor.get_chains():
                    receptor.__dict__[seq].metadata.custom_params["cluster_id"] = str(labels[index])
                receptor.metadata["cluster_id"] = str(labels[index])

            processed_dataset = ReceptorDataset.build_from_objects(receptors=processed_receptors, file_size=dataset.file_size, name=dataset.name, path=path)
            processed_dataset.encoded_data = dataset.encoded_data
            processed_dataset.labels = dataset.labels
        else:
            from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
            processed_sequences = [x for x in dataset.get_data()]
            for index, seq in enumerate(processed_sequences):
                seq.metadata.custom_params["cluster_id"] = str(labels[index])

            processed_dataset = SequenceDataset.build_from_objects(sequences=processed_sequences, file_size=dataset.file_size, name=dataset.name, path=path)
            processed_dataset.encoded_data = dataset.encoded_data
            processed_dataset.labels = dataset.labels

        processed_dataset.encoded_data.labels["cluster_id"] = labels
        processed_dataset.labels["cluster_id"] = list(range(max(labels) + 1))
        print_log("Finished copying dataset.", include_datetime=True)
        return processed_dataset

    @staticmethod
    def _copy_if_exists(old_file: Path, path: Path):
        import shutil
        if old_file is not None and old_file.is_file():
            new_file = path / old_file.name
            if not new_file.is_file():
                shutil.copyfile(old_file, new_file)
            return new_file
        else:
            raise RuntimeError(f"Clustering instruction: tried exporting file {old_file}, but it does not exist.")
