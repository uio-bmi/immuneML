# quality: peripheral
from importlib import import_module

import yaml

from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingStrategy import SequenceEncodingStrategy
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from source.encodings.word2vec.model_creator.ModelType import ModelType
from source.ml_methods.LogisticRegression import LogisticRegression
from source.ml_methods.RandomForestClassifier import RandomForestClassifier
from source.ml_methods.SVM import SVM
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.IdentityMotifInstantiation import IdentityMotifInstantiation
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


class Parser:
    """
    Simple DSL parser from python dictionary or equivalent YAML for configuring repertoire / receptor_sequence
    classification in the (simulated) settings
    """

    @staticmethod
    def parse_yaml_file(file_path) -> dict:
        with open(file_path, "r") as file:
            workflow_specification = yaml.load(file)

        return Parser.parse(workflow_specification)

    @staticmethod
    def parse(workflow_specification: dict) -> dict:
        result = {}
        if "simulation" in workflow_specification:
            result["simulation"], result["signals"] = Parser._parse_simulation(workflow_specification["simulation"])
        if "ml_methods" in workflow_specification:
            result["ml_methods"] = Parser._parse_ml_methods(workflow_specification["ml_methods"])
        if "encoder" in workflow_specification and "encoder_params" in workflow_specification:
            result["encoder"], result["encoder_params"] = Parser._parse_encoder(workflow_specification)

        for key in workflow_specification.keys():
            if key not in result.keys():
                result[key] = workflow_specification[key]

        if "batch_size" not in workflow_specification.keys():
            result["batch_size"] = 1  # TODO: read this from configuration file / config object or sth similar

        return result

    @staticmethod
    def _parse_encoder(workflow_specification: dict):
        if workflow_specification["encoder"] == "KmerFrequencyEncoder":
            assert "sequence_encoding" in workflow_specification["encoder_params"], "Parser: creating encoder: sequence_encoding for KmerFrequencyEncoder is not specified."
            encoder = KmerFrequencyEncoder()
            encoder_params = Parser._parse_encoder_params(workflow_specification["encoder_params"], encoder)
        else:
            assert "model" in workflow_specification["encoder_params"] and "model_creator" in workflow_specification["encoder_params"]["model"], "Parser: creating encoder: model_creator for Word2VecEncoder is not specified."
            encoder = Word2VecEncoder()
            encoder_params = Parser._parse_encoder_params(workflow_specification["encoder_params"], encoder)

        return encoder, encoder_params

    @staticmethod
    def _parse_encoder_params(encoder_params: dict, encoder: DatasetEncoder) -> dict:
        parsed_encoder_params = {}

        if isinstance(encoder, KmerFrequencyEncoder):
            parsed_encoder_params["sequence_encoding_strategy"] = Parser._transform_sequence_encoding_strategy(encoder_params["sequence_encoding"])
            parsed_encoder_params["reads"] = ReadsType.UNIQUE if encoder_params["reads"] == "unique" else ReadsType.ALL
            parsed_encoder_params["normalization_type"] = NormalizationType.L2 if encoder_params["normalization_type"] == "l2" else NormalizationType.RELATIVE_FREQUENCY
        elif isinstance(encoder, Word2VecEncoder):
            parsed_encoder_params["model"] = {
                "k": encoder_params["model"]["k"],
                "size": encoder_params["model"]["size"],
                "model_creator": ModelType.SEQUENCE if encoder_params["model"]["model_creator"] == "receptor_sequence" else ModelType.KMER_PAIR
            }

        for key in encoder_params.keys():
            if key not in parsed_encoder_params.keys():
                parsed_encoder_params[key] = encoder_params[key]

        return parsed_encoder_params

    @staticmethod
    def _transform_sequence_encoding_strategy(sequence_encoding_strategy: str) -> SequenceEncodingStrategy:
        val = getattr(SequenceEncodingType, sequence_encoding_strategy.upper()).value
        (module_path, _, class_name) = val.rpartition(".")
        module = import_module(module_path)
        sequence_encoding_strategy_instance = getattr(module, class_name)()
        return sequence_encoding_strategy_instance

    @staticmethod
    def _build_default_methods(ml_methods: list) -> list:
        methods = []

        for key in ml_methods:
            mod = import_module("source.ml_methods.{}".format(key))
            method = getattr(mod, key)()
            methods.append(method)

        return methods

    @staticmethod
    def _build_methods_with_params(ml_methods: dict) -> list:
        methods = []
        for key in ml_methods.keys():

            param_grid = {k: ml_methods[key][k] if isinstance(ml_methods[key][k], list) else [ml_methods[key][k]]
                          for k in ml_methods[key].keys()}

            mod = import_module("source.ml_methods.{}".format(key))
            method = getattr(mod, key)(parameter_grid=param_grid)
            methods.append(method)

        return methods

    @staticmethod
    def _parse_ml_methods(ml_methods) -> list:

        if isinstance(ml_methods, list):
            methods = Parser._build_default_methods(ml_methods)
        else:
            methods = Parser._build_methods_with_params(ml_methods)

        return methods

    @staticmethod
    def _parse_simulation(simulation: dict):
        assert "motifs" in simulation, "Workflow specification parser: no motifs were defined for the simulation."
        assert "signals" in simulation, "Workflow specification parser: no signals were defined for the simulation."

        motifs = Parser._extract_motifs(simulation)
        signals = Parser._extract_signals(simulation, motifs)
        implanting = Parser._add_signals_to_implanting(simulation, signals)

        return implanting, signals

    @staticmethod
    def _add_signals_to_implanting(simulation: dict, signals: list) -> list:
        result = []
        for item in simulation["implanting"]:
            result.append({
                "repertoires": item["repertoires"],
                "sequences": item["sequences"],
                "signals": [signal for signal in signals if signal.id in item["signals"]]
            })
        return result

    @staticmethod
    def _extract_motifs(simulation: dict) -> list:
        motifs = []
        for item in simulation["motifs"]:
            instantiation_strategy = Parser._get_instantiation_strategy(item)
            motif = Motif(item["id"], instantiation_strategy, item["seed"])
            motifs.append(motif)
        return motifs

    @staticmethod
    def _extract_signals(simulation: dict, motifs: list) -> list:
        signals = []
        for item in simulation["signals"]:
            implanting_strategy = Parser._get_implanting_strategy(item)
            signal_motifs = [motif for motif in motifs if motif.id in item["motifs"]]
            signal = Signal(item["id"], signal_motifs, implanting_strategy)
            signals.append(signal)
        return signals

    @staticmethod
    def _get_implanting_strategy(signal: dict) -> SignalImplantingStrategy:
        if "implanting" in signal and signal["implanting"] == "healthy_sequences":
            implanting_strategy = HealthySequenceImplanting(GappedMotifImplanting())
        else:
            raise NotImplementedError
        return implanting_strategy

    @staticmethod
    def _get_instantiation_strategy(motif_item: dict) -> MotifInstantiationStrategy:
        if "instantiation" in motif_item and motif_item["instantiation"] == "identity":
            instantiation_strategy = IdentityMotifInstantiation()
        else:
            raise NotImplementedError
        return instantiation_strategy
