import logging

import numpy as np

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.SequenceType import SequenceType


class PositionHelper:
    MAX_CDR3_LEN = 91
    MIN_CDR3_LEN = 5
    MIDPOINT_CDR3_LEN = 13

    @staticmethod
    def get_imgt_position_weights_for_annotation(input_length: int, region_type: RegionType,
                                                 sequence_position_weights: dict):
        imgt_positions = PositionHelper.gen_imgt_positions_from_length(input_length, region_type)

        position_weights = {}
        if sequence_position_weights:
            for index, position in enumerate(imgt_positions):
                if position in sequence_position_weights:
                    position_weights[position] = sequence_position_weights[position]

        if len(imgt_positions) > len(position_weights):
            weights_sum = sum(list(position_weights.values()))
            remaining_weight_for_position = (1 - weights_sum) / (len(imgt_positions) - len(position_weights))
            for position in imgt_positions:
                if position not in position_weights:
                    position_weights[position] = remaining_weight_for_position

        if not np.isclose(sum(list(position_weights.values())), 1):
            weights_sum = sum(list(position_weights.values()))
            position_weights = {position: weight / weights_sum for position, weight in position_weights.items()}

        return {position: position_weights[position] for position in imgt_positions}

    @staticmethod
    def get_allowed_positions_for_annotation(input_length: int, region_type: RegionType,
                                             sequence_position_weights: dict):
        position_weights = PositionHelper.get_imgt_position_weights_for_annotation(input_length, region_type,
                                                                                   sequence_position_weights)
        return [int(bool(weight)) for weight in position_weights.values()]

    @staticmethod
    def get_imgt_position_weights_for_implanting(input_length: int, region_type: RegionType,
                                                 sequence_position_weights: dict, limit: int):
        position_weights = PositionHelper.get_imgt_position_weights_for_annotation(input_length, region_type,
                                                                                   sequence_position_weights)

        for index, position in enumerate(position_weights.keys()):
            if index > input_length - limit:
                position_weights[position] = 0.

        weights_sum = sum(list(position_weights.values()))
        if weights_sum == 0:
            logging.warning(f"Sequence of length {input_length} has no allowed positions for signal with sequence "
                            f"position weights {sequence_position_weights} and motif length {limit}, it will be discarded.")
            return position_weights

        position_weights = {position: np.array([weight]).astype(np.float64)[0] / weights_sum
                            for position, weight in position_weights.items()}

        assert np.isclose(sum(list(position_weights.values())), 1.), \
            (input_length, region_type.name, position_weights, sum(list(position_weights.values())), limit)

        return position_weights

    @staticmethod
    def gen_imgt_positions_from_cdr3_length(input_length: int) -> list:
        if PositionHelper.MIN_CDR3_LEN <= input_length <= PositionHelper.MIDPOINT_CDR3_LEN:
            positions = [105, 106, 107, 116, 117]
            pos_left_count = (input_length - PositionHelper.MIN_CDR3_LEN) // 2
            pos_right_count = input_length - PositionHelper.MIN_CDR3_LEN - pos_left_count

            positions = ([str(pos) for pos in positions if pos <= 107] +
                         [str(i) for i in range(108, 107 + pos_left_count + 1)]
                         + [str(i) for i in range(116 - pos_right_count, 116)] + ['116', '117'])
            return positions

        elif PositionHelper.MIDPOINT_CDR3_LEN < input_length <= PositionHelper.MAX_CDR3_LEN:
            positions = list(range(105, 118))
            pos111_count = (input_length - PositionHelper.MIDPOINT_CDR3_LEN) // 2
            pos112_count = input_length - PositionHelper.MIDPOINT_CDR3_LEN - pos111_count

            positions = ([str(pos) for pos in positions if pos <= 111] +
                         [f'111.{i}' for i in range(1, pos111_count + 1)]
                         + [f'112.{i}' for i in range(pos112_count, 0, -1)] +
                         [str(pos) for pos in positions if pos >= 112])

            return positions
        else:
            logging.warning(f"IMGT positions could not be generated for CDR3 sequence of length {input_length}.")
            return []

    @staticmethod
    def gen_imgt_positions_from_junction_length(input_length: int):
        if PositionHelper.MIN_CDR3_LEN + 2 <= input_length <= PositionHelper.MAX_CDR3_LEN + 2:
            return ['104'] + PositionHelper.gen_imgt_positions_from_cdr3_length(input_length - 2) + ['118']
        else:
            logging.warning(
                f"IMGT positions could not be generated for IMGT junction sequence of length {input_length}.")
            return []

    @staticmethod
    def gen_imgt_positions_from_sequence(sequence: ReceptorSequence,
                                         sequence_type: SequenceType = SequenceType.AMINO_ACID):
        if sequence_type != SequenceType.AMINO_ACID:
            raise NotImplementedError(f"{sequence_type.name} is currently not supported for obtaining IMGT positions")
        region_type = sequence.get_attribute("region_type")
        input_length = len(sequence.get_sequence(sequence_type=sequence_type))

        return PositionHelper.gen_imgt_positions_from_length(input_length, region_type)

    @staticmethod
    def gen_imgt_positions_from_length(input_length: int, region_type: RegionType):
        if region_type == RegionType.IMGT_CDR3:
            return PositionHelper.gen_imgt_positions_from_cdr3_length(input_length)
        if region_type == RegionType.IMGT_JUNCTION:
            return PositionHelper.gen_imgt_positions_from_junction_length(input_length)
        else:
            raise NotImplementedError(
                f"PositionHelper: IMGT positions are not implemented for region type {region_type}")
