import os
import shutil
from unittest import TestCase

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.LogisticRegression import LogisticRegression
from immuneML.reports.ml_reports.ConfounderAnalysis import ConfounderAnalysis
from immuneML.simulation.Implanting import Implanting
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationState import SimulationState
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.steps.SignalImplanter import SignalImplanter


class TestConfounderAnalysis(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_lr_model(self, path, encoded_data, label):
        # dummy logistic regression with 100 observations with 3 features belonging to 2 classes
        dummy_lr = LogisticRegression()
        dummy_lr.fit_by_cross_validation(encoded_data,
                                         number_of_splits=2, label_name=label)

        return dummy_lr

    def _make_dataset(self, path, size) -> RepertoireDataset:

        random_dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=size, sequence_count_probabilities={100: 1.},
                                                                            sequence_length_probabilities={5: 1.}, labels={}, path=path)

        signals = [Signal(identifier="disease", motifs=[Motif(identifier="m1", instantiation=GappedKmerInstantiation(), seed="AAA")],
                          implanting_strategy=HealthySequenceImplanting(implanting_computation=ImplantingComputation.ROUND,
                                                                        implanting=GappedMotifImplanting())),
                   Signal(identifier="HLA", motifs=[Motif(identifier="m2", instantiation=GappedKmerInstantiation(), seed="CCC")],
                          implanting_strategy=HealthySequenceImplanting(implanting_computation=ImplantingComputation.ROUND,
                                                                        implanting=GappedMotifImplanting())),
                   Signal(identifier="age", motifs=[Motif(identifier="m3", instantiation=GappedKmerInstantiation(), seed="GGG")],
                          implanting_strategy=HealthySequenceImplanting(implanting_computation=ImplantingComputation.ROUND,
                                                                        implanting=GappedMotifImplanting()))]

        simulation = Simulation([Implanting(dataset_implanting_rate=0.2, signals=signals, name='i1', repertoire_implanting_rate=0.25),
                                 Implanting(dataset_implanting_rate=0.2, signals=[signals[0], signals[1]], name='i2', repertoire_implanting_rate=0.25),
                                 Implanting(dataset_implanting_rate=0.1, signals=[signals[0]], name='i3', repertoire_implanting_rate=0.25),
                                 Implanting(dataset_implanting_rate=0.2, signals=[signals[2]], name='i4', repertoire_implanting_rate=0.25),
                                 Implanting(dataset_implanting_rate=0.1, signals=[signals[1]], name='i5', repertoire_implanting_rate=0.25)
                                 ])

        dataset = SignalImplanter.run(SimulationState(signals=signals, dataset=random_dataset, formats=['ImmuneML'], result_path=path,
                                                      name='my_synthetic_dataset', simulation=simulation))

        return dataset

    def _encode_dataset(self, encoder, dataset, path, learn_model: bool = True):
        # encodes the repertoire by frequency of 3-mers
        lc = LabelConfiguration()
        lc.add_label("disease", [True, False])
        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=learn_model,
            model={}
        ))
        return encoded_dataset

    def _create_report(self, path):
        report = ConfounderAnalysis.build_object(metadata_labels=["age", "HLA"], name='test')

        report.ml_details_path = path / "ml_details.yaml"
        report.label = "disease"
        report.result_path = path
        encoder = KmerFrequencyEncoder.build_object(RepertoireDataset(), **{
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "k": 3,
            'sequence_type': SequenceType.AMINO_ACID.name
        })
        report.train_dataset = self._encode_dataset(encoder, self._make_dataset(path / "train", size=100), path)
        report.test_dataset = self._encode_dataset(encoder, self._make_dataset(path / "test", size=40), path, learn_model=False)
        report.method = self._create_dummy_lr_model(path, report.train_dataset.encoded_data, report.label)

        return report

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "confounder_report/"
        PathBuilder.build(path)

        report = self._create_report(path)

        # Running the report
        result = report._generate()

        shutil.rmtree(path)
