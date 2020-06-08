from unittest import TestCase

from source.util.FilenameHandler import FilenameHandler


class TestFilenameHandler(TestCase):
    def test_get_filename(self):
        name = FilenameHandler.get_filename("RandomForestClassifier", "json")
        self.assertEqual("random_forest_classifier.json", name)

    def test_get_dataset_name(self):
        name = FilenameHandler.get_dataset_name("KmerFrequencyEncoder")
        self.assertEqual("encoded_dataset.iml_dataset", name)

    def test_model_name(self):
        name = FilenameHandler.get_model_name("Word2VecEncoder")
        self.assertEqual("word2_vec_encoder_model.pickle", name)
