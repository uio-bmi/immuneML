import logging
import math

from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence


class PositionHelper:
    @staticmethod
    def gen_imgt_positions_from_length(input_length: int):
        start = 105
        end = 117
        imgt_range = list(range(start, end + 1))
        length = input_length if input_length < 14 else 13
        imgt_positions = imgt_range[:math.ceil(length / 2)] + imgt_range[-math.floor(length / 2):]
        if input_length > 13:
            len_insert = input_length - 13
            insert_left = [111 + 0.001 * i for i in range(1, math.floor(len_insert / 2) + 1)]
            insert_right = [112 + 0.001 * i for i in range(1, math.ceil(len_insert / 2) + 1)]
            insert = insert_left + list(reversed(insert_right))
            imgt_positions[math.ceil(len(imgt_range) / 2):math.ceil(len(imgt_range) / 2)] = insert
        return imgt_positions

    @staticmethod
    def gen_imgt_positions_from_sequence(sequence: ReceptorSequence):
        input_length = len(sequence.get_sequence())
        return PositionHelper.gen_imgt_positions_from_length(input_length)

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

        position_weights = {int(imgt_positions[k]): sequence_position_weights[imgt_positions[k]]
                            if imgt_positions[k] in sequence_position_weights.keys() and k < index_limit else 0.0 for k
                            in range(len(imgt_positions))}
        weights_sum = sum([position_weights[k] for k in sequence_position_weights.keys() if k in position_weights])
        # normalize weights
        if weights_sum != 0:
            position_weights = {int(k): float(position_weights[k]) / float(weights_sum) for k in position_weights.keys()}
        else:
            position_weights = {int(k): 1 / len(position_weights.keys()) for k in position_weights}
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
