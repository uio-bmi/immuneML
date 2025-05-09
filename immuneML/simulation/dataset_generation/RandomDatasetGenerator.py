import copy
import random
import uuid
from pathlib import Path

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.SequenceParams import RegionType, ChainPair
from immuneML.data_model.SequenceSet import ReceptorSequence, Receptor
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class RandomDatasetGenerator:

    @staticmethod
    def _check_probabilities(probabilities_dict, key_type, dict_name):
        assert isinstance(probabilities_dict, dict) and all(
            isinstance(key, key_type) for key in probabilities_dict.keys()) \
               and all(isinstance(value, float) or value in {0, 1} for value in probabilities_dict.values()) \
               and 0.99 <= round(sum(probabilities_dict.values()), 5) <= 1, \
            f"RandomDatasetGenerator: {dict_name} are not specified correctly. They should be a dictionary with probabilities per count " \
            f"and sum to 1, but got {probabilities_dict} instead."

    @staticmethod
    def _check_labels(labels: dict):
        if labels is not None:
            assert isinstance(labels, dict)
            for label in labels:
                RandomDatasetGenerator._check_probabilities(labels[label], object, f"labels - {label}")

    @staticmethod
    def _check_example_count(count, name):
        assert isinstance(count, int) and count > 0, f"RandomDatasetGenerator: {name} is not specified properly. " \
                                                     f"It should be a positive integer, got {count} instead."

    @staticmethod
    def _check_path(path: Path):
        assert path is not None, "RandomDatasetGenerator: path cannot be None when generating datasets."

    @staticmethod
    def _check_rep_dataset_generation_params(repertoire_count: int, sequence_count_probabilities: dict,
                                             sequence_length_probabilities: dict,
                                             labels: dict, path: Path):

        RandomDatasetGenerator._check_example_count(repertoire_count, "repertoire_count")
        RandomDatasetGenerator._check_probabilities(sequence_count_probabilities, int, "sequence_count_probabilities")
        RandomDatasetGenerator._check_probabilities(sequence_length_probabilities, int, "sequence_length_probabilities")
        RandomDatasetGenerator._check_labels(labels)
        RandomDatasetGenerator._check_path(path)

    @staticmethod
    def generate_repertoire_dataset(repertoire_count: int, sequence_count_probabilities: dict,
                                    sequence_length_probabilities: dict,
                                    labels: dict, path: Path, name="repertoire_dataset") -> RepertoireDataset:
        """
        Creates repertoire_count repertoires where the number of sequences per repertoire is sampled from the probability distribution given
        in sequence_count_probabilities. The length of sequences is sampled independently for each sequence from
        sequence_length_probabilities distribution. The labels are also randomly assigned to repertoires from the distribution given in
        labels. In this case, labels are multi-class, so each repertoire will get at one class from each label. This means that negative
        classes for the labels should be included as well in the specification.

        An example of input parameters is given below:
        repertoire_count: 100 # generate 100 repertoires
        sequence_count_probabilities:
            100: 0.5 # half of the generated repertoires will have 100 sequences
            200: 0.5 # the other half of the generated repertoires will have 200 sequences
        sequence_length_distribution:
            14: 0.8 # 80% of all generated sequences for all repertoires will have length 14
            15: 0.2 # 20% of all generated sequences across all repertoires will have length 15
        labels:
            cmv: # label name
                True: 0.5 # 50% of the repertoires will have class True
                False: 0.5 # 50% of the repertoires will have class False
            coeliac: # next label with classes that will be assigned to repertoires independently of the previous label or any other parameter
                1: 0.3 # 30% of the generated repertoires will have class 1
                0: 0.7 # 70% of the generated repertoires will have class 0
        """
        RandomDatasetGenerator._check_rep_dataset_generation_params(repertoire_count, sequence_count_probabilities,
                                                                    sequence_length_probabilities,
                                                                    labels, path)

        alphabet = EnvironmentSettings.get_sequence_alphabet()
        PathBuilder.build(path)

        sequences = [["".join(random.choices(alphabet,
                                             k=random.choices(list(sequence_length_probabilities.keys()),
                                                              sequence_length_probabilities.values())[0]))
                      for seq_count in range(
                random.choices(list(sequence_count_probabilities.keys()), sequence_count_probabilities.values())[0])]
                     for rep in range(repertoire_count)]

        processed_labels, dataset_params = RandomDatasetGenerator._make_labels(labels, repertoire_count)

        dataset = RepertoireBuilder.build_dataset(sequences=sequences, path=path, labels=processed_labels, name=name)

        AIRRExporter.export(dataset, path)

        return dataset

    @staticmethod
    def _make_labels(labels: dict, element_count: int):
        if labels is not None:
            processed_labels = {
                label: random.choices(list(labels[label].keys()), labels[label].values(), k=element_count) for label
                in labels}
            dataset_params = {label: list(labels[label].keys()) for label in labels}
        else:
            processed_labels = None
            dataset_params = None

        return processed_labels, dataset_params

    @staticmethod
    def _check_receptor_dataset_generation_params(receptor_count: int, chain_1_length_probabilities: dict,
                                                  chain_2_length_probabilities: dict, labels: dict, path: Path):

        RandomDatasetGenerator._check_probabilities(chain_1_length_probabilities, int, 'chain_1_length_probabilities')
        RandomDatasetGenerator._check_probabilities(chain_2_length_probabilities, int, 'chain_2_length_probabilities')
        RandomDatasetGenerator._check_example_count(receptor_count, "receptor_count")
        RandomDatasetGenerator._check_labels(labels)
        RandomDatasetGenerator._check_path(path)

    @staticmethod
    def generate_receptor_dataset(receptor_count: int, chain_1_length_probabilities: dict,
                                  chain_2_length_probabilities: dict, labels: dict,
                                  path: Path, name="receptor_dataset"):
        """
        Creates receptor_count receptors where the length of sequences in each chain is sampled independently for each sequence from
        chain_n_length_probabilities distribution. The labels are also randomly assigned to receptors from the distribution given in
        labels. In this case, labels are multi-class, so each receptor will get one class from each label. This means that negative
        classes for the labels should be included as well in the specification. chain 1 and 2 in this case refer to alpha and beta
        chain of a T-cell receptor.

        An example of input parameters is given below:

        receptor_count: 100 # generate 100 TRABReceptors
        chain_1_length_probabilities:
            14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
            15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
        chain_2_length_probabilities:
            14: 0.8 # 80% of all generated sequences for all receptors (for chain 2) will have length 14
            15: 0.2 # 20% of all generated sequences across all receptors (for chain 2) will have length 15
        labels:
            epitope1: # label name
                True: 0.5 # 50% of the receptors will have class True
                False: 0.5 # 50% of the receptors will have class False
            epitope2: # next label with classes that will be assigned to receptors independently of the previous label or other parameters
                1: 0.3 # 30% of the generated receptors will have class 1
                0: 0.7 # 70% of the generated receptors will have class 0
        """
        RandomDatasetGenerator._check_receptor_dataset_generation_params(receptor_count, chain_1_length_probabilities,
                                                                         chain_2_length_probabilities, labels, path)

        alphabet = EnvironmentSettings.get_sequence_alphabet()
        PathBuilder.build(path)

        get_random_sequence = lambda proba, chain, id: ReceptorSequence(
            sequence_aa="".join(random.choices(alphabet, k=random.choices(list(proba.keys()), proba.values())[0])),
            duplicate_count=random.randint(1, 10), v_call=chain + "V1-1*01", j_call=chain + "J1-1*01", locus=chain, cell_id=str(id))

        receptors = []
        for i in range(receptor_count):
            receptor_id = uuid.uuid4().hex
            receptors.append(Receptor(chain_1=get_random_sequence(chain_1_length_probabilities, "TRA", receptor_id),
                                      chain_2=get_random_sequence(chain_2_length_probabilities, "TRB", receptor_id),
                                      chain_pair=ChainPair["TRA_TRB"],
                                      cell_id=str(receptor_id), receptor_id=receptor_id,
                                      metadata={
                                          **{label: random.choices(list(label_dict.keys()), label_dict.values(), k=1)[0]
                                             for label, label_dict in labels.items()}, **{"subject": f"subj_{i + 1}"}}))

        for receptor in receptors:
            receptor.chain_1.metadata = copy.deepcopy(receptor.metadata)
            receptor.chain_2.metadata = copy.deepcopy(receptor.metadata)

        processed_labels, dataset_params = RandomDatasetGenerator._make_labels(labels, receptor_count)
        dataset = ReceptorDataset.build_from_objects(receptors=receptors, path=path, name=name,
                                                     labels=dataset_params)

        AIRRExporter.export(dataset, path)

        return dataset

    @staticmethod
    def _check_sequence_dataset_generation_params(receptor_count: int, length_probabilities: dict, labels: dict,
                                                  path: Path):
        RandomDatasetGenerator._check_probabilities(length_probabilities, int, 'length_probabilities')
        RandomDatasetGenerator._check_example_count(receptor_count, "receptor_count")
        RandomDatasetGenerator._check_labels(labels)
        RandomDatasetGenerator._check_path(path)

    @staticmethod
    def generate_sequence_dataset(sequence_count: int, length_probabilities: dict, labels: dict, path: Path,
                                  region_type: str = RegionType.IMGT_CDR3.name, name="sequence_dataset"):
        """
        Creates sequence_count receptor sequences (single chain) where the length of sequences in each chain is sampled independently for each sequence from
        length_probabilities distribution. The labels are also randomly assigned to sequences from the distribution given in
        labels. In this case, labels are multi-class, so each sequences will get one class from each label. This means that negative
        classes for the labels should be included as well in the specification.

        An example of input parameters is given below:

        sequence_count: 100 # generate 100 TRB ReceptorSequences
        length_probabilities:
            14: 0.8 # 80% of all generated sequences for all receptors (for chain 1) will have length 14
            15: 0.2 # 20% of all generated sequences across all receptors (for chain 1) will have length 15
        labels:
            epitope1: # label name
                True: 0.5 # 50% of the receptors will have class True
                False: 0.5 # 50% of the receptors will have class False
            epitope2: # next label with classes that will be assigned to receptors independently of the previous label or other parameters
                1: 0.3 # 30% of the generated receptors will have class 1
                0: 0.7 # 70% of the generated receptors will have class 0
        """
        RandomDatasetGenerator._check_sequence_dataset_generation_params(sequence_count, length_probabilities, labels,
                                                                         path)

        alphabet = EnvironmentSettings.get_sequence_alphabet()
        PathBuilder.build(path)

        chain = "TRB"

        sequences = [ReceptorSequence(uuid.uuid4().hex, None,
                                      "".join(random.choices(alphabet,
                                                             k=random.choices(list(length_probabilities.keys()),
                                                                              length_probabilities.values())[0])),
                                      locus=chain, v_call=chain + "V1-1*01", j_call=chain + "J1-1*01",
                                      duplicate_count=random.randint(1, 10),
                                      metadata={
                                          **{label: random.choices(list(label_dict.keys()), label_dict.values(), k=1)[0]
                                             for label, label_dict in labels.items()}, **{"subject": f"subj_{i + 1}"}})
                     for i in range(sequence_count)]

        processed_labels, dataset_params = RandomDatasetGenerator._make_labels(labels, sequence_count)
        dataset = SequenceDataset.build_from_objects(sequences, path, name=name, labels=dataset_params,
                                                     region_type=RegionType[region_type.upper()])

        AIRRExporter.export(dataset, path)

        return dataset
