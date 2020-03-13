import copy
import os
import pickle

from sklearn.pipeline import make_pipeline

from source.IO.dataset_export.PickleExporter import PickleExporter
from source.IO.dataset_import.PickleLoader import PickleLoader
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.definition_parsers.EncodingParser import EncodingParser
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.EncoderParams import EncoderParams
from source.util.ReflectionHandler import ReflectionHandler


class PipelineEncoder(DatasetEncoder):
    """
    Encodes the dataset using an initial encoder and then passes it through a pipeline of
    steps to modify this initial encoding. This can be useful for feature selection and feature
    summarization as well as annotation of biological data onto the initially encoded dataset.

    Arguments:
        initial_encoder (DatasetEncoder):
        initial_encoder_params (dict):
        steps (list):

    Specification:
        initial_encoder: KmerFrequency
        initial_encoder_params: {k: 3}
        steps:
            - annotate_sequences:
                # type can be the name of any class which inherits TransformerMixin class from scikit-learn
                # custom immuneML classes which do this are located under encodings/pipeline/steps/
                type: SequenceMatchFeatureAnnotation
                params:
                    reference_sequence_path: reference_sequence_path
                    data_loader_params:
                        result_path: ./path/
                        dataset_id: dataset_id
                        additional_columns: ["Antigen Protein"]
                        strip_CF: True
                        column_mapping:
                            amino_acid: "CDR3B AA Sequence"
                            v_gene: "TRBV Gene"
                            j_gene: "TRBJ Gene"
                    sequence_matcher_params:
                        max_distance: 0
                        metadata_fields_to_match: []
                        same_length: True
                    data_loader_name: GenericLoader
                    annotation_prefix: annotation_prefix
    """

    def __init__(self, initial_encoder, initial_encoder_params, steps: list):
        self.initial_encoder, self.initial_encoder_params, _ = EncodingParser.parse_encoder_internal(initial_encoder, initial_encoder_params)
        self.steps = PipelineEncoder._prepare_steps(steps)

    @staticmethod
    def _prepare_steps(steps: list):
        parsed_steps = []
        for step in steps:
            for key in step:
                step_class = ReflectionHandler.get_class_by_name(step[key]["type"])
                parsed_steps.append(step_class(**step[key].get("params", {})))
        assert len(steps) == len(parsed_steps), "PipelineParser: Each step accepts only one specification."
        return parsed_steps

    @staticmethod
    def build_object(dataset, **params):
        if isinstance(dataset, RepertoireDataset):
            return PipelineEncoder(**params if params is not None else {})
        else:
            raise ValueError("PipelineEncoder is not defined for dataset types which are not RepertoireDataset.")

    def encode(self, dataset, params: EncoderParams):
        filepath = params["result_path"] + "/" + params["filename"]
        if os.path.isfile(filepath):
            encoded_dataset = self._run_pipeline(PickleLoader.load(filepath), params)
        else:
            encoded_dataset = self._encode_new_dataset(dataset, params)
        return encoded_dataset

    def _encode_new_dataset(self, dataset, params: EncoderParams):
        inital_encoded_dataset = self._initial_encode_examples(dataset, params)
        encoded_dataset = self._run_pipeline(inital_encoded_dataset, params)
        self.store(encoded_dataset, params)
        return encoded_dataset

    def _initial_encode_examples(self, dataset, params: EncoderParams):
        initial_params = EncoderParams(
            result_path=params["result_path"],
            label_configuration=params["label_configuration"],
            batch_size=params["batch_size"],
            learn_model=params["learn_model"],
            filename=params["filename"],
            model=None
        )
        encoder = self.initial_encoder.build_object(dataset, **self.initial_encoder_params)
        encoded_dataset = encoder.encode(dataset, initial_params)
        return encoded_dataset

    def _run_pipeline(self, dataset, params: EncoderParams):
        pipeline_file = params["result_path"] + "Pipeline.pickle"
        steps = self.extend_steps(params)
        if params["learn_model"]:
            pipeline = make_pipeline(*steps)
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

    def extend_steps(self, params: EncoderParams):
        steps = copy.deepcopy(self.steps)
        for index, step in enumerate(steps):
            step.result_path = params["result_path"]
            step.filename = params["filename"]
            step.initial_encoder = self.initial_encoder.__class__.__name__
            step.initial_params = tuple((key, self.initial_encoder_params[key])
                                        for key in self.initial_encoder_params.keys())
            step.previous_steps = self._prepare_previous_steps(steps, index)
        return steps

    def _prepare_previous_steps(self, steps, index):
        return tuple(step.to_tuple() for i, step in enumerate(steps) if i < index)

    def store(self, encoded_dataset, params: EncoderParams):
        PickleExporter.export(encoded_dataset, params["result_path"], params["filename"])
