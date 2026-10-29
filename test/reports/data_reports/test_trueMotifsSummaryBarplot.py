import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model import bnp_util
from immuneML.data_model.SequenceParams import RegionType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.data_reports.TrueMotifsSummaryBarplot import TrueMotifsSummaryBarplot
from immuneML.util.PathBuilder import PathBuilder

TRAIN_SEQUENCE = "CASSLGTEAFF"
ONE_SUBSTITUTION = "CASSLGTEAFY"
TWO_SUBSTITUTIONS = "CASSLGTEAYY"
UNRELATED = "CAWSVGDTQYW"


class TestTrueMotifsSummaryBarplot(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _get_plotting_data(self):
        sequence_column = bnp_util.get_sequence_field_name(RegionType.IMGT_CDR3, SequenceType.AMINO_ACID)
        return pd.DataFrame({
            sequence_column: [TRAIN_SEQUENCE, "CASSPQRGYTF", TRAIN_SEQUENCE, ONE_SUBSTITUTION,
                              TWO_SUBSTITUTIONS, UNRELATED],
            'data_origin': ['original_train', 'original_train', 'PWM', 'PWM', 'PWM', 'PWM']})

    def _build_report(self, path, max_edit_distance, compairr_path=None, show_test_set_baseline=False):
        return TrueMotifsSummaryBarplot(implanted_motifs_per_signal={}, result_path=path,
                                        region_type=RegionType.IMGT_CDR3, max_edit_distance=max_edit_distance,
                                        compairr_path=compairr_path,
                                        show_test_set_baseline=show_test_set_baseline)

    def test_default_params_are_loadable(self):
        defaults = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/",
                                            "true_motifs_summary_barplot")

        # the motifs have no natural default and must be given in the specification
        self.assertNotIn("implanted_motifs_per_signal", defaults)
        self.assertEqual(defaults["max_edit_distance"], 0)
        self.assertFalse(defaults["show_test_set_baseline"])

        report = TrueMotifsSummaryBarplot.build_object(**{**defaults, "implanted_motifs_per_signal": {}})
        self.assertEqual(report.max_edit_distance, 0)
        self.assertFalse(report.show_test_set_baseline)

    def test_test_set_baseline(self):
        compairr_path = next((p for p in EnvironmentSettings.compairr_paths if p.exists()), None)
        assert compairr_path is not None, "CompAIRR binary not found; build it to run this test"

        path = EnvironmentSettings.tmp_test_path / "true_motifs_summary_baseline/"
        PathBuilder.remove_old_and_build(path)

        plotting_data = self._get_plotting_data()
        plotting_data.loc[3, 'data_origin'] = 'original_test'

        # by default the held-out sequences are not scored at all
        report = self._build_report(path, max_edit_distance=2, compairr_path=str(compairr_path))
        labelled = report._add_novelty_memorization_label(plotting_data.copy())
        self.assertEqual(labelled.loc[3, 'novelty_memorization'], 'original')

        # as the baseline they are scored against the training set exactly like the generated sequences
        report = self._build_report(path, max_edit_distance=2, compairr_path=str(compairr_path),
                                    show_test_set_baseline=True)
        labelled = report._add_novelty_memorization_label(plotting_data.copy())
        self.assertEqual(labelled.loc[3, 'novelty_memorization'], 'within 1 edit')
        self.assertListEqual(list(labelled['novelty_memorization']),
                             ['original', 'original', 'memorized', 'within 1 edit', 'within 2 edits', 'novel'])

        shutil.rmtree(path)

    def test_exact_matching_only(self):
        path = EnvironmentSettings.tmp_test_path / "true_motifs_summary_exact/"
        PathBuilder.remove_old_and_build(path)

        report = self._build_report(path, max_edit_distance=0)
        labelled = report._add_novelty_memorization_label(self._get_plotting_data())

        # without an edit distance, one substitution is enough to count as novel
        self.assertListEqual(list(labelled['novelty_memorization']),
                             ['original', 'original', 'memorized', 'novel', 'novel', 'novel'])

        shutil.rmtree(path)

    def test_near_copies_with_compairr(self):
        compairr_path = next((p for p in EnvironmentSettings.compairr_paths if p.exists()), None)
        assert compairr_path is not None, "CompAIRR binary not found; build it to run this test"

        path = EnvironmentSettings.tmp_test_path / "true_motifs_summary_near_copies/"
        PathBuilder.remove_old_and_build(path)

        report = self._build_report(path, max_edit_distance=2, compairr_path=str(compairr_path))
        labelled = report._add_novelty_memorization_label(self._get_plotting_data())

        # each generated sequence gets the smallest distance at which it matches a training sequence
        self.assertListEqual(list(labelled['novelty_memorization']),
                             ['original', 'original', 'memorized', 'within 1 edit', 'within 2 edits', 'novel'])

        shutil.rmtree(path)

    def test_near_copies_stop_at_max_edit_distance(self):
        compairr_path = next((p for p in EnvironmentSettings.compairr_paths if p.exists()), None)
        assert compairr_path is not None, "CompAIRR binary not found; build it to run this test"

        path = EnvironmentSettings.tmp_test_path / "true_motifs_summary_one_edit/"
        PathBuilder.remove_old_and_build(path)

        report = self._build_report(path, max_edit_distance=1, compairr_path=str(compairr_path))
        labelled = report._add_novelty_memorization_label(self._get_plotting_data())

        # the sequence two substitutions away stays novel when only one edit is allowed
        self.assertListEqual(list(labelled['novelty_memorization']),
                             ['original', 'original', 'memorized', 'within 1 edit', 'novel', 'novel'])

        shutil.rmtree(path)

    def test_motif_recovery_summary(self):
        path = EnvironmentSettings.tmp_test_path / "true_motifs_summary_table/"
        PathBuilder.remove_old_and_build(path)

        annotated = pd.DataFrame({
            'data_origin': ['original_train', 'original_train', 'original_test', 'PWM', 'PWM', 'PWM', 'PWM'],
            'novelty_memorization': ['original', 'original', 'original', 'memorized', 'novel', 'novel', 'novel'],
            'signal1': [1, 0, 1, 1, 1, 0, 0],
            'signal2': [0, 1, 0, 0, 1, 1, 0]})

        report = TrueMotifsSummaryBarplot(implanted_motifs_per_signal={'signal1': {}, 'signal2': {}},
                                          result_path=path)
        summary = report._get_motif_recovery_summary(annotated)

        # original_test is left out unless it is shown as the baseline, as in the figure
        self.assertSetEqual(set(summary['data_origin'].astype(str)), {'original_train', 'PWM'})

        pwm_novel = summary[(summary['data_origin'] == 'PWM') & (summary['novelty_memorization'] == 'novel')]
        pwm_novel = pwm_novel.set_index('signal')
        # a sequence carrying both signals is counted once for each of them
        self.assertEqual(pwm_novel.loc['signal1', 'count'], 1)
        self.assertEqual(pwm_novel.loc['signal2', 'count'], 2)
        self.assertAlmostEqual(pwm_novel.loc['signal2', 'frequency'], 2 / 4)
        self.assertEqual(pwm_novel.loc['signal1', 'total_sequences'], 4)
        self.assertAlmostEqual(pwm_novel.loc['signal1', 'signal_specific_percent'], 75.)

        report = TrueMotifsSummaryBarplot(implanted_motifs_per_signal={'signal1': {}, 'signal2': {}},
                                          result_path=path, show_test_set_baseline=True)
        summary = report._get_motif_recovery_summary(annotated)
        # the originals come first, the held-out baseline next to them, then the models
        self.assertListEqual(list(dict.fromkeys(summary['data_origin'].astype(str))),
                             ['original_train', 'original_test', 'PWM'])

        shutil.rmtree(path)

    def test_without_training_sequences(self):
        path = EnvironmentSettings.tmp_test_path / "true_motifs_summary_no_train/"
        PathBuilder.remove_old_and_build(path)

        plotting_data = self._get_plotting_data()
        plotting_data['data_origin'] = 'PWM'

        report = self._build_report(path, max_edit_distance=2)
        with self.assertLogs(level="WARNING"):
            labelled = report._add_novelty_memorization_label(plotting_data)

        # nothing to compare against, so the exact-match labels are kept rather than the report failing
        self.assertListEqual(list(labelled['novelty_memorization']), ['novel'] * 6)

        shutil.rmtree(path)
