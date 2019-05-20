import datetime
import shutil
import warnings

from source.data_model.dataset.Dataset import Dataset
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.dsl.semantic_model.Blackboard import Blackboard
from source.encodings.EncoderParams import EncoderParams
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder
from source.workflows.processes.MLProcess import MLProcess
from source.workflows.steps.DataEncoder import DataEncoder


class SemanticModel:

    def __init__(self, path: str = None, specification_path: str = None):
        self._encoding_connections = {}  # e1: d1, e2: d2 -> encoding_id: dataset_id
        self._ml_method_connections = {}  # ml1: {encoding: e1, assessment_type: LOOCV} -> ml_id: {encoding: enc_id, assessment_type: str}
        self._report_connections = {}  # r1: {encoding: e1}, r2: {ml_method: ml1} -> report_id: {encoding: enc_id, ml_method: ml_id, dataset: dataset_id}
        self._executed = set()
        self._symbol_table = None
        self._path = path if path is not None else EnvironmentSettings.default_analysis_path
        self._specification_path = specification_path
        self._blackboard = Blackboard()

    def fill(self, symbol_table: SymbolTable):
        self._symbol_table = symbol_table

        self.add_connections(self.add_report_connection, ["encoding", "ml_method", "dataset"], SymbolType.REPORT)
        self.add_connections(self.add_ml_connection, ["encoding"], SymbolType.ML_METHOD)
        self.add_connections(self.add_encoding_connection, ["dataset"], SymbolType.ENCODING)

    def execute(self):
        if self._is_ready():
            self._execute()
        else:
            warnings.warn("Semantic model execution was attempted, "
                          "but the dependencies were not fully specified. Returning instead...")

    def _execute(self):
        self._execute_reports()
        self._execute_ml_methods()

    def _execute_reports(self):
        self._execute_dataset_reports()
        self._execute_encoding_reports()
        self._execute_ml_reports()

    def _execute_ml_methods(self):
        for method_id in self._ml_method_connections.keys():
            self._execute_ml_method(method_id)

    def _execute_ml_method(self, method_id):

        if method_id not in self._executed:

            method = self._symbol_table.get(method_id)
            dataset = self._symbol_table.get(self._symbol_table.get(method["encoding"])["dataset"])["dataset"]
            label_config = LabelConfiguration()
            for label in method["labels"]:
                label_config.add_label(label, dataset.params[label])

            ml_path = self._create_result_path("machine_learning_{}".format(method_id))
            self._copy_specification_to_result(ml_path)

            proc = MLProcess(dataset=dataset,
                             split_count=method["split_count"],
                             path=ml_path,
                             label_configuration=label_config,
                             encoder=self._symbol_table.get(method["encoding"])["encoder"],
                             encoder_params=self._symbol_table.get(method["encoding"])["params"],
                             method=method["method"],
                             assessment_type=method["assessment_type"],
                             metrics=method["metrics"],
                             model_selection_cv=method["model_selection_cv"],
                             model_selection_n_folds=method["model_selection_n_folds"],
                             min_example_count=method["min_example_count"])
            proc.run()
            self._update_executed(method_id)

    def _execute_dataset_reports(self):
        self._execute_reports_by_type("dataset")

    def _execute_encoding_reports(self):
        self._execute_reports_by_type("encoding")

    def _execute_ml_reports(self):
        self._execute_reports_by_type("ml_method")

    def _execute_reports_by_type(self, key_to_check: str):
        report_ids = [key for key in self._report_connections.keys()
                      if key_to_check in self._report_connections[key].keys()]

        reports = [self._symbol_table.get(key) for key in report_ids]

        # TODO: parallelize this

        for index, report in enumerate(reports):
            if report_ids[index] not in self._executed:
                self._execute_prerequisites(self._report_connections[report_ids[index]])
                params = self._prepare_report_params(report_ids[index], report["params"])
                self._copy_specification_to_result(params["result_path"])
                report["report"].generate_report(params)
                self._update_executed(report_ids[index])

    def _execute_prerequisites(self, prerequisites: dict):
        if "ml_method" in prerequisites and prerequisites["ml_method"] is not None:
            self._execute_ml_method(prerequisites["ml_method"])

        if "encoding" in prerequisites and prerequisites["encoding"] is not None:
            self._execute_encoding(prerequisites["encoding"])

    def _execute_encoding(self, encoding_id: str):
        if encoding_id not in self._executed:
            encoder_dict = self._symbol_table.get(encoding_id)
            dataset = self._symbol_table.get(encoder_dict["dataset"])["dataset"]
            path = self._create_result_path("encoding_{}".format(encoding_id))
            label_config = self._prepare_label_config(encoder_dict["labels"], dataset)

            encoded = self._encode_dataset(dataset, encoder_dict["encoder"], label_config, encoder_dict["params"], path)
            self._blackboard.add("{}_{}".format(encoder_dict["dataset"], encoding_id), encoded)
            self._update_executed(encoding_id)

    def _copy_specification_to_result(self, result_path):
        PathBuilder.build(result_path)
        shutil.copy(self._specification_path, result_path)

    def _is_ready(self):
        """
        checks if necessary dependencies are defined for the model (e.g. if report is given, if the required encoding
        has a defined dataset), but does not check if those ids actually point to sth in the symbol table -> that's up
        to the parser to fill in
        :return: whether all dependencies where defined for the model and if symbol table has been set
        """
        reports_ready = self._check_connections(self._report_connections, self._encoding_connections, "encoding") \
                        and self._check_connections(self._report_connections, self._ml_method_connections, "ML_method")

        ml_methods_ready = self._check_connections(self._ml_method_connections, self._encoding_connections, "encoding")

        return reports_ready and ml_methods_ready and isinstance(self._symbol_table, SymbolTable)

    def _check_connections(self, items: dict, dependency_items: dict, key: str):
        ready = True
        i = 0
        keys = list(items.keys())
        while ready and i < len(keys):
            if key in items.keys():
                ready = keys[i] in dependency_items.keys()
            i += 1
        return ready

    def _encode_dataset(self, dataset, encoder, label_config, params, path) -> Dataset:
        return DataEncoder.run({
            "dataset": dataset,
            "encoder": encoder,
            "encoder_params": EncoderParams(result_path=self._create_result_path("encoding"),
                                            filename="dataset.pickle",
                                            label_configuration=label_config,
                                            model=params, batch_size=4, learn_model=True,
                                            model_path=path, scaler_path=path, vectorizer_path=path, pipeline_path=path)
        })

    def _prepare_label_config(self, labels, dataset):
        label_config = LabelConfiguration()
        for label in labels:
            label_config.add_label(label, dataset.params[label])
        return label_config

    def _update_executed(self, key: str):
        self._executed.add(key)
        if key in self._report_connections.keys() and "ml_method" in self._report_connections[key]:
            self._executed.add(self._report_connections[key]["ml_method"])

    def _prepare_report_params(self, report_id: str, initial_params: dict) -> dict:
        params = {"result_path": self._create_result_path("report_{}".format(report_id)),
                  "dataset": self._symbol_table.get(initial_params["dataset"])["dataset"]
                  if "dataset" in initial_params else None}
        if "encoding" in initial_params.keys():
            params["dataset"] = self._blackboard.get("{}_{}".format(self._symbol_table
                                                                    .get(initial_params["encoding"])["dataset"],
                                                                    initial_params["encoding"]))
        params = {**initial_params, **params}
        return params

    def _create_result_path(self, result_type: str):
        now = datetime.datetime.now()
        print("storing results in {}".format(self._path + "{}_{}/".format(result_type, str(now).replace(".", "_").replace(":", "_"))))
        return self._path + "{}_{}/".format(result_type, str(now).replace(".", "_").replace(":", "_"))


    def add_connections(self, fn, keys, symbol_type: SymbolType):
        for item_id, item in self._symbol_table.get_by_type(symbol_type):
            existing_keys = [key for key in keys if key in item.keys()]
            fn(item_id, {key: item[key] for key in existing_keys})

    def add_encoding_connection(self, encoding_id: str, connection: dict):
        assert "dataset" in connection.keys(), \
            "Dependency is not properly defined between the encoding and the dataset."
        self._encoding_connections[encoding_id] = connection

    def add_ml_connection(self, ml_id: str, connection: dict):
        assert "encoding" in connection.keys(), \
            "Dependency is not properly defined between the machine learning model {} and the encoding.".format(ml_id)
        self._ml_method_connections[ml_id] = connection

    def add_report_connection(self, report_id: str, connection: dict):
        assert any([key in connection.keys() for key in ["encoding", "dataset", "ml_method"]]), \
            "Dependency is not properly defined for the report, input is not declared."
        self._report_connections[report_id] = connection
