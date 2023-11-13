from unittest import TestCase

import numpy as np

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.util.PositionHelper import PositionHelper


def test_get_imgt_position_weights_from_annotation():
    weights = PositionHelper.get_imgt_position_weights_for_annotation(len("CASSSKASTDTQYF"), RegionType.IMGT_JUNCTION,
                                                                      {})

    assert np.isclose(sum(list(weights.values())), 1.), weights
    weights_values = list(weights.values())
    assert all(weights_values[0] == val for val in weights_values), weights

    weights = PositionHelper.get_imgt_position_weights_for_annotation(len("CASSSKASTDTQYF"), RegionType.IMGT_JUNCTION,
                                                                      {'104': 0, '105': 0, '110': 0})

    assert np.isclose(sum(list(weights.values())), 1.), weights
    assert weights['104'] == 0 and weights['105'] == 0 and weights['110'] == 0, weights


def test_get_imgt_position_weights_for_implanting():
    weights = PositionHelper.get_imgt_position_weights_for_implanting(7, RegionType.IMGT_JUNCTION,
                                                                      {'104': 0, '105': 0}, 2)

    assert np.isclose(sum(list(weights.values())), 1), weights
    assert all(weights[pos] == 0 for pos in ['104', '105', '118'])
    assert all(np.isclose(weights[pos], 0.25) for pos in ['106', '107', '116', '117'])

    weights = PositionHelper.get_imgt_position_weights_for_implanting(len("CASSSKASTDTQYF"), RegionType.IMGT_JUNCTION,
                                                                      {}, limit=len("KTC"))

    assert np.isclose(sum(list(weights.values())), 1), weights
    assert all(weight == 0 for pos, weight in weights.items() if pos in ['117', '118']), weights
    assert weights['116'] > 0, weights

    weights = PositionHelper.get_imgt_position_weights_for_implanting(34, RegionType.IMGT_JUNCTION,
                                                                      {'104': 0, '105': 0, '118': 0}, 2)

    assert np.isclose(sum(list(weights.values())), 1), weights


def test_get_allowed_positions_for_annotation():
    allowed_positions = PositionHelper.get_allowed_positions_for_annotation(34, RegionType.IMGT_JUNCTION, {})
    assert len(allowed_positions) == 34
