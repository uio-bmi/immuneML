# quality: peripheral
from source.encodings.DatasetEncoder import DatasetEncoder
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
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
    Simple DSL parser from python dictionary for configuring repertoire / sequence classification
    in the (simulated) settings
    """
    @staticmethod
    def parse(workflow_specification: dict) -> dict:
        result = {}
        if "simulation" in workflow_specification:
            result["simulation"], result["signals"] = Parser.__parse_simulation(workflow_specification["simulation"])
        if "ml_methods" in workflow_specification:
            result["ml_methods"] = Parser.__parse_ml_methods(workflow_specification["ml_methods"])
        if "encoder" in workflow_specification and "encoder_params" in workflow_specification:
            result["encoder"], result["encoder_params"] = Parser.__parse_encoder(workflow_specification)

        for key in workflow_specification.keys():
            if key not in result.keys():
                result[key] = workflow_specification[key]

        if "batch_size" not in workflow_specification.keys():
            result["batch_size"] = 1  # TODO: read this from configuration file / config object or sth similar

        return result

    @staticmethod
    def __parse_encoder(workflow_specification: dict):
        if workflow_specification["encoder"] == "KmerFrequencyEncoder":
            assert "sequence_encoding" in workflow_specification["encoder_params"], "Parser: creating encoder: sequence_encoding for KmerFrequencyEncoder is not specified."
            encoder = KmerFrequencyEncoder()
            encoder_params = Parser.__parse_encoder_params(workflow_specification["encoder_params"], encoder)
        else:
            assert "model" in workflow_specification["encoder_params"] and "model_creator" in workflow_specification["encoder_params"]["model"], "Parser: creating encoder: model_creator for Word2VecEncoder is not specified."
            encoder = Word2VecEncoder()
            encoder_params = Parser.__parse_encoder_params(workflow_specification["encoder_params"], encoder)

        return encoder, encoder_params

    @staticmethod
    def __parse_encoder_params(encoder_params: dict, encoder: DatasetEncoder) -> dict:
        parsed_encoder_params = {}

        if isinstance(encoder, KmerFrequencyEncoder):
            parsed_encoder_params["sequence_encoding_strategy"] = Parser.__transform_sequence_encoding_strategy(encoder_params["sequence_encoding"])
            parsed_encoder_params["reads"] = ReadsType.UNIQUE if encoder_params["reads"] == "unique" else ReadsType.ALL
            parsed_encoder_params["normalization_type"] = NormalizationType.L2 if encoder_params["normalization_type"] == "l2" else NormalizationType.RELATIVE_FREQUENCY
        elif isinstance(encoder, Word2VecEncoder):
            parsed_encoder_params["model"] = {
                "k": encoder_params["model"]["k"],
                "size": encoder_params["model"]["size"],
                "model_creator": ModelType.SEQUENCE if encoder_params["model"]["model_creator"] == "sequence" else ModelType.KMER_PAIR
            }

        for key in encoder_params.keys():
            if key not in parsed_encoder_params.keys():
                parsed_encoder_params[key] = encoder_params[key]

        return parsed_encoder_params

    @staticmethod
    def __transform_sequence_encoding_strategy(sequence_encoding_strategy: str) -> SequenceEncodingType:
        if sequence_encoding_strategy == "gapped_kmer":
            sequence_encoding_type = SequenceEncodingType.GAPPED_KMER
        elif sequence_encoding_strategy == "IMGT_gapped_kmer":
            sequence_encoding_type = SequenceEncodingType.IMGT_GAPPED_KMER
        elif sequence_encoding_strategy == "IMGT_continuous_kmer":
            sequence_encoding_type = SequenceEncodingType.IMGT_CONTINUOUS_KMER
        elif sequence_encoding_strategy == "continuous_kmer":
            sequence_encoding_type = SequenceEncodingType.CONTINUOUS_KMER
        else:
            sequence_encoding_type = SequenceEncodingType.IDENTITY

        return sequence_encoding_type

    @staticmethod
    def __parse_ml_methods(ml_methods: list) -> list:

        methods = []

        if "LogisticRegression" in ml_methods:
            methods.append(LogisticRegression())
        if "SVM" in ml_methods:
            methods.append(SVM())
        if "RandomForest" in ml_methods:
            methods.append(RandomForestClassifier())

        return methods

    @staticmethod
    def __parse_simulation(simulation: dict):
        assert "motifs" in simulation, "Workflow specification parser: no motifs were defined for the simulation."
        assert "signals" in simulation, "Workflow specification parser: no signals were defined for the simulation."

        motifs = Parser.__extract_motifs(simulation)
        signals = Parser.__extract_signals(simulation, motifs)
        implanting = Parser.__add_signals_to_implanting(simulation, signals)

        return implanting, signals

    @staticmethod
    def __add_signals_to_implanting(simulation: dict, signals: list) -> list:
        result = []
        for item in simulation["implanting"]:
            result.append({
                "repertoires": item["repertoires"],
                "sequences": item["sequences"],
                "signals": [signal for signal in signals if signal.id in item["signals"]]
            })
        return result

    @staticmethod
    def __extract_motifs(simulation: dict) -> list:
        motifs = []
        for item in simulation["motifs"]:
            instantiation_strategy = Parser.__get_instantiation_strategy(item)
            motif = Motif(item["id"], instantiation_strategy, item["seed"])
            motifs.append(motif)
        return motifs

    @staticmethod
    def __extract_signals(simulation: dict, motifs: list) -> list:
        signals = []
        for item in simulation["signals"]:
            implanting_strategy = Parser.__get_implanting_strategy(item)
            signal_motifs = [motif for motif in motifs if motif.id in item["motifs"]]
            signal = Signal(item["id"], signal_motifs, implanting_strategy)
            signals.append(signal)
        return signals

    @staticmethod
    def __get_implanting_strategy(signal: dict) -> SignalImplantingStrategy:
        if "implanting" in signal and signal["implanting"] == "healthy_sequences":
            implanting_strategy = HealthySequenceImplanting(GappedMotifImplanting())
        else:
            raise NotImplementedError
        return implanting_strategy

    @staticmethod
    def __get_instantiation_strategy(motif_item: dict) -> MotifInstantiationStrategy:
        if "instantiation" in motif_item and motif_item["instantiation"] == "identity":
            instantiation_strategy = IdentityMotifInstantiation()
        else:
            raise NotImplementedError
        return instantiation_strategy
