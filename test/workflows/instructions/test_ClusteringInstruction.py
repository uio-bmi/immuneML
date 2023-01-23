import os
import shutil
from unittest import TestCase

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.util.PathBuilder import PathBuilder
from immuneML.ml_methods.Clustering.KMeans import KMeans
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.util.ReadsType import ReadsType
from immuneML.util.RepertoireBuilder import RepertoireBuilder
from immuneML.workflows.instructions.clustering.ClusteringInstruction import ClusteringInstruction
from immuneML.workflows.instructions.clustering.ClusteringUnit import ClusteringUnit


class TestClusteringProcess(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dataset(self, path):
        repertoires, metadata = RepertoireBuilder.build([["AAA"], ["AAAC"], ["ACA"], ["CAAA"], ["AAAC"], ["AAA"]], path,
                                                        {"l1": [1, 1, 1, 0, 0, 0], "l2": [2, 3, 2, 3, 2, 3]})

        dataset = RepertoireDataset(repertoires=repertoires, labels={"l1": [0, 1], "l2": [2, 3]}, metadata_file=metadata)
        return dataset

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "clusteringprocess/"
        PathBuilder.build(path)

        dataset = self.create_dataset(path)

        label_config = LabelConfiguration()
        label_config.add_label("l1", [0, 1])
        label_config.add_label("l2", [2, 3])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
        100a	TRA	AAAC	TRAV12	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	                    
        """

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        refs_dict = {"path": path / "refs.tsv", "format": "VDJdb"}

        clusteringAlgo = KMeans()
        encoder = KmerFrequencyEncoder.build_object(dataset, **{
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "k": 3,
            'sequence_type': SequenceType.AMINO_ACID.name
        })

        units = {"named_analysis_1": ClusteringUnit(dataset=dataset, report=SequenceLengthDistribution(), clustering_method=clusteringAlgo, encoder=encoder, number_of_processes=16),
                 "named_analysis_2": ClusteringUnit(dataset=dataset, report=SequenceLengthDistribution(), clustering_method=clusteringAlgo, encoder=encoder)}
        process = ClusteringInstruction(units, name="clusteringProc")
        process.run(path / "results/")

        self.assertTrue(units["named_analysis_1"].number_of_processes == 16)
        self.assertTrue(os.path.isfile(path / "results/exp/analysis_named_analysis_1/report/sequence_length_distribution.html"))
        self.assertTrue(os.path.isfile(path / "results/exp/analysis_named_analysis_2/report/sequence_length_distribution.html"))

        shutil.rmtree(path)
