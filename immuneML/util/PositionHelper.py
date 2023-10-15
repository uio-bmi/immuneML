import logging

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.environment.SequenceType import SequenceType


class PositionHelper:
    MAX_CDR3_LEN = 91
    MIN_CDR3_LEN = 5
    MIDPOINT_CDR3_LEN = 13

    @staticmethod
    def gen_imgt_positions_from_cdr3_length(input_length: int):
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
    def gen_imgt_positions_from_sequence(sequence: ReceptorSequence, sequence_type: SequenceType = SequenceType.AMINO_ACID):
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
            raise NotImplementedError(f"PositionHelper: IMGT positions are not implemented for region type {region_type}")

    @staticmethod
    def adjust_position_weights(sequence_position_weights: dict, imgt_positions, limit: int) -> dict:
        """
        :param sequence_position_weights: weights supplied by the user as to where in the receptor_sequence to implant
        :param imgt_positions: IMGT positions present in the specific receptor_sequence
        :param limit: how far from the end of the receptor_sequence the motif at latest must start
                        in order not to elongate the receptor_sequence
        :return: position_weights for implanting a motif instance into a receptor_sequence
        """
        # filter only position weights where there are imgt positions in the receptor_sequence and 0 if this imgt position is
        # not in the sequence_position_weights
        index_limit = len(imgt_positions) - limit

        position_weights = {imgt_positions[k]: sequence_position_weights[imgt_positions[k]]
                            if imgt_positions[k] in sequence_position_weights.keys() and k < index_limit else 0.0 for k
                            in range(len(imgt_positions))}
        weights_sum = sum([position_weights[k] for k in sequence_position_weights.keys() if k in position_weights])
        # normalize weights
        if weights_sum != 0:
            position_weights = {k: float(position_weights[k]) / float(weights_sum) for k in position_weights.keys()}
        else:
            position_weights = {k: 1 / len(position_weights.keys()) for k in position_weights}
        return position_weights

    @staticmethod
    def build_position_weights(sequence_position_weights: dict, imgt_positions, limit: int) -> dict:
        if sequence_position_weights is not None:
            position_weights = PositionHelper.adjust_position_weights(sequence_position_weights, imgt_positions, limit)
        else:
            valid_position_count = len(imgt_positions) - limit
            position_weights = {imgt_positions[i]: 1.0 / valid_position_count if i < valid_position_count else 0
                                for i in range(len(imgt_positions))}
            logging.warning('Position weights are not defined. Randomly choosing position to implant motif_instance instead.')

        return position_weights
