import shutil
from pprint import pprint
from unittest import TestCase

from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.BCReceptor import BCReceptor
from immuneML.data_model.receptor.ElementGenerator import ElementGenerator
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestElementGenerator(TestCase):
    def test_build_batch_generator(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "element_batch_generator/")

        receptors = [BCReceptor(identifier=str(i),
                                heavy=ReceptorSequence('A', metadata=SequenceMetadata(locus='HEAVY', cell_id=str(i))),
                                light=ReceptorSequence('C', metadata=SequenceMetadata(locus='LIGHT', cell_id=str(i))))
                     for i in range(307)]
        file_list = [path / f"batch{i+1}.tsv" for i in range(4)]

        dataset = ReceptorDataset.build_from_objects(receptors, 100, path)

        receptor_generator = ElementGenerator(file_list, element_class_name=BCReceptor.__name__,
                                              buffer_type=dataset.element_generator.buffer_type)
        generator = receptor_generator.build_batch_generator()

        counter = 0

        for batch in generator:
            for receptor in batch:
                self.assertEqual(counter, int(receptor.identifier))
                self.assertTrue(isinstance(receptor, BCReceptor))
                counter += 1

        self.assertEqual(307, counter)

        generator = receptor_generator.build_batch_generator()

        counter = 0

        for batch in generator:
            for receptor in batch:
                self.assertEqual(counter, int(receptor.identifier))
                self.assertTrue(isinstance(receptor, BCReceptor))
                counter += 1

        self.assertEqual(307, counter)

        shutil.rmtree(path)

    def make_seq_dataset(self, path, seq_count: int = 100, file_size: int = 10) -> SequenceDataset:
        sequences = []
        for i in range(seq_count):
            sequences.append(ReceptorSequence(sequence_aa="AAA", sequence_id=str(i)))

        return SequenceDataset.build_from_objects(sequences, file_size, path, 'dataset_name1')

    def make_rec_dataset(self, path, count: int, file_size: int) -> ReceptorDataset:
        receptors = []
        for i in range(count):
            receptors.append(TCABReceptor(ReceptorSequence("AA", metadata=SequenceMetadata(locus='alpha', cell_id=str(i))),
                                          ReceptorSequence('CCC', metadata=SequenceMetadata(locus='beta', cell_id=str(i))),
                                          identifier=str(i)))

        return ReceptorDataset.build_from_objects(receptors, file_size, path, 'dataset_rec1')

    def test_make_subset(self):

        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "element_generator_subset/")
        d = self.make_seq_dataset(path)

        indices = [1, 20, 21, 22, 23, 24, 25, 50, 52, 60, 70, 77, 78, 90, 92]

        d2 = d.make_subset(indices, path, SequenceDataset.TRAIN)

        for batch in d2.get_batch(1000):
            for sequence in batch:
                self.assertTrue(int(sequence.sequence_id) in indices)

        self.assertEqual(15, d2.get_example_count())

        shutil.rmtree(path)

    def test_get_data_from_index_range(self):
        for ind, dataset_gen_func in enumerate([self.make_seq_dataset, self.make_rec_dataset]):
            path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / f"el_gen_index_range_{ind}/")

            dataset = dataset_gen_func(path, 18, 5)
            elements = dataset.get_data_from_index_range(7, 13)
            pprint(elements)
            assert len(elements) == 7

            shutil.rmtree(path)

