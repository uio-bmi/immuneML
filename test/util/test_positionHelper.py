from unittest import TestCase

from immuneML.util.PositionHelper import PositionHelper


class TestPositionHelper(TestCase):
    def test_gen_imgt_positions_from_length(self):
        length = 11
        positions = PositionHelper.gen_imgt_positions_from_cdr3_length(length)
        self.assertEqual(positions, ['105', '106', '107', '108', '109', '110', '113', '114', '115', '116', '117'])
        length = 12
        positions = PositionHelper.gen_imgt_positions_from_cdr3_length(length)
        self.assertEqual(positions, ['105', '106', '107', '108', '109', '110', '112', '113', '114', '115', '116', '117'])
        length = 13
        positions = PositionHelper.gen_imgt_positions_from_cdr3_length(length)
        self.assertEqual(positions, ['105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117'])
        length = 18
        positions = PositionHelper.gen_imgt_positions_from_cdr3_length(length)
        self.assertEqual(positions,
                         ['105', '106', '107', '108', '109', '110', '111', '111.1', '111.2', '112.3', '112.2', '112.1', '112', '113', '114',
                          '115', '116', '117'])

    def test_adjust_position_weights(self):
        weights = PositionHelper.adjust_position_weights(sequence_position_weights={105: 0.8, 106: 0.1, 109: 0.1},
                                                         imgt_positions=[105, 106, 107, 108, 109, 113, 114, 115],
                                                         limit=5)

        self.assertEqual(0.89, round(weights[105], 2))
        self.assertEqual(0.11, round(weights[106], 2))
        self.assertTrue(all([weights[key] == 0 for key in weights.keys() if key > 106]))

    def test_build_position_weights(self):
        weights = PositionHelper.build_position_weights(None, [105, 106, 107, 108, 114, 115], 1)
        self.assertEqual(0.2, weights[105])
        self.assertEqual(0.2, weights[106])
        self.assertEqual(0.2, weights[107])
        self.assertEqual(0.2, weights[108])
        self.assertEqual(0.2, weights[114])
        self.assertEqual(0, weights[115])

        weights = PositionHelper.build_position_weights({105: 0.5}, [105, 106, 107, 108, 114, 115], 1)
        self.assertEqual(1, weights[105])
        self.assertEqual(0, weights[106])
        self.assertEqual(0, weights[107])
        self.assertEqual(0, weights[108])
        self.assertEqual(0, weights[114])
        self.assertEqual(0, weights[115])

        weights = PositionHelper.build_position_weights({105: 0.3, 106: 0.4, 107: 0.3, 115: 0.3}, [105, 106, 107, 108, 114, 115], 1)
        self.assertEqual(0.3, weights[105])
        self.assertEqual(0.4, weights[106])
        self.assertEqual(0.3, weights[107])
        self.assertEqual(0, weights[108])
        self.assertEqual(0, weights[114])
        self.assertEqual(0, weights[115])
