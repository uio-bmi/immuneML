import re


class StringHelper:

    @staticmethod
    def camel_case_to_words(camel_case_string: str):
        string = camel_case_string[0].upper() + camel_case_string[1:]
        return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)

    @staticmethod
    def camel_case_to_word_string(camel_case_string: str):
        return " ".join(StringHelper.camel_case_to_words(camel_case_string))

    @staticmethod
    def pad_sequence_in_the_middle(sequence: str, max_len: int, pad_char: str) -> str:
        if len(sequence) == max_len:
            return sequence
        else:
            pad_start = len(sequence) // 2
            pad_len = max_len - len(sequence)
            return sequence[:pad_start] + pad_char * pad_len + sequence[pad_start:]
