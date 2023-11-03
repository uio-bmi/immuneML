from unittest import TestCase

from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.hyperparameter_optimization.strategy.GridSearch import GridSearch
from immuneML.ml_methods.classifiers.LogisticRegression import LogisticRegression


class TestGridSearch(TestCase):
    def test_generate_next_setting(self):

        hp_settings = [HPSetting(encoder=KmerFrequencyEncoder, encoder_params={}, encoder_name="enc1", ml_method=LogisticRegression(),
                                 ml_params={"model_selection_cv": False, "model_selection_n_fold": -1}, ml_method_name="ml1",
                                 preproc_sequence=[]),
                       HPSetting(encoder=Word2VecEncoder, encoder_params={}, encoder_name="enc2", ml_method=LogisticRegression(),
                                 ml_params={"model_selection_cv": False, "model_selection_n_fold": -1}, ml_method_name="ml2",
                                 preproc_sequence=[])]

        grid_search = GridSearch(hp_settings)
        setting1 = grid_search.generate_next_setting()
        setting2 = grid_search.generate_next_setting(setting1, 0.7)
        setting3 = grid_search.generate_next_setting(setting2, 0.8)

        self.assertIsNone(setting3)
        self.assertEqual(KmerFrequencyEncoder, setting1.encoder)
        self.assertEqual(Word2VecEncoder, setting2.encoder)

    def test_get_optimal_hps(self):
        hp_settings = [HPSetting(encoder=KmerFrequencyEncoder, encoder_params={}, encoder_name="e1", ml_method=LogisticRegression(),
                                 ml_params={"model_selection_cv": False, "model_selection_n_fold": -1}, ml_method_name="ml1",
                                 preproc_sequence=[]),
                       HPSetting(encoder=Word2VecEncoder, encoder_params={}, encoder_name='e2', ml_method=LogisticRegression(),
                                 ml_params={"model_selection_cv": False, "model_selection_n_fold": -1}, ml_method_name="ml2",
                                 preproc_sequence=[])]

        grid_search = GridSearch(hp_settings)
        setting1 = grid_search.generate_next_setting()
        setting2 = grid_search.generate_next_setting(setting1, 0.7)
        grid_search.generate_next_setting(setting2, 0.8)

        optimal = grid_search.get_optimal_hps()

        self.assertEqual(hp_settings[1], optimal)
