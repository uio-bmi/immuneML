import os
import pickle

from sklearn.pipeline import make_pipeline

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams


class PipelineEncoder(DatasetEncoder):
    """
    Encodes the dataset using an initial encoder and then passes it thorough a pipeline of
    steps to modify this initial encoding. Can be useful for feature selection and feature
    summarization as well as annotation of biological data onto the initially encoded dataset.

    Configuration parameters are an instance of EncoderParams:
    {
        "model": {
            "initial_encoder": initial_encoder,
            "initial_encoder_params": initial_encoder_params (can just be a dictionary, will be converted
            into EncoderParams class later),
            "steps": steps (details below)
        },
        "batch_size": 1,
        "learn_model": True, # true for training set and false for test set
        "result_path": "../",
        "label_configuration": LabelConfiguration(), # labels should be set before encodings is invoked,
        "model_path": "../",
        "scaler_path": "../",
        "vectorizer_path": None
    }

    encoder_params["model"]["steps"] must be in the following format:
    [
        {
            "annotate_sequences":
                {
                    "type": "SequenceMatchFeatureAnnotation",
                    "params":
                        {
                            "reference_sequence_path": "reference_sequence_path",
                            "data_loader_params":
                            {
                                "result_path": "result_path",
                                "dataset_id": "dataset_id",
                                "additional_columns": ["Antigen Protein"],
                                "strip_CF": True,
                                "column_mapping": {
                                    "amino_acid": "CDR3B AA Sequence",
                                    "v_gene": "TRBV Gene",
                                    "j_gene": "TRBJ Gene"
                                }
                            },
                            "sequence_matcher_params":
                            {
                                "max_distance": 0,
                                "metadata_fields_to_match": [],
                                "same_length": True
                            },
                            "data_loader_name": "GenericLoader",
                            "annotation_prefix": "annotation_prefix"
                        }
                },
        },
    ...
    ]

    Some examples of workflows can be seen in the PipelineEncoder integration tests.
    """

    @staticmethod
    def encode(dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        filepath = params["result_path"] + "/" + params["filename"]
        if os.path.isfile(filepath):
            encoded_dataset = PipelineEncoder._run_pipeline(PickleLoader.load(filepath), params)
        else:
            encoded_dataset = PipelineEncoder._encode_new_dataset(dataset, params)
        return encoded_dataset

    @staticmethod
    def _encode_new_dataset(dataset: RepertoireDataset, params: EncoderParams) -> RepertoireDataset:
        inital_encoded_dataset = PipelineEncoder._initial_encode_repertoires(dataset, params)
        encoded_dataset = PipelineEncoder._run_pipeline(inital_encoded_dataset, params)
        PipelineEncoder.store(encoded_dataset, params)
        return encoded_dataset

    @staticmethod
    def _initial_encode_repertoires(dataset: RepertoireDataset, params: EncoderParams):
        initial_params = EncoderParams(
            result_path=params["result_path"],
            label_configuration=params["label_configuration"],
            batch_size=params["batch_size"],
            learn_model=params["learn_model"],
            filename=params["filename"],
            model=params["model"]["initial_encoder_params"]
        )
        encoded_dataset = params["model"]["initial_encoder"].encode(dataset, initial_params)
        return encoded_dataset

    @staticmethod
    def _run_pipeline(dataset: RepertoireDataset, params: EncoderParams):
        pipeline_file = params["result_path"] + "Pipeline.pickle"
        params["model"] = PipelineEncoder.extend_steps(params)
        if params["learn_model"]:
            pipeline = make_pipeline(*params["model"]["steps"])
            encoded_dataset = pipeline.fit_transform(dataset)
            with open(pipeline_file, 'wb') as file:
                pickle.dump(pipeline, file)
        else:
            with open(pipeline_file, 'rb') as file:
                pipeline = pickle.load(file)
            for step in pipeline.steps:
                step[1].result_path = params["result_path"]
                step[1].filename = params["filename"]
            encoded_dataset = pipeline.transform(dataset)

        return encoded_dataset

    @staticmethod
    def extend_steps(params: EncoderParams):
        for index, step in enumerate(params["model"]["steps"]):
            step.result_path = params["result_path"]
            step.filename = params["filename"]
            step.initial_encoder = params["model"]["initial_encoder"].__class__.__name__
            step.initial_params = tuple((key, params["model"]["initial_encoder_params"][key])
                                         for key in params["model"]["initial_encoder_params"].keys())
            step.previous_steps = PipelineEncoder._prepare_previous_steps(params["model"], index)
        return params["model"]

    @staticmethod
    def _prepare_previous_steps(model, index):
        return tuple(step.to_tuple() for i, step in enumerate(model["steps"]) if i < index)

    @staticmethod
    def store(encoded_dataset: RepertoireDataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
