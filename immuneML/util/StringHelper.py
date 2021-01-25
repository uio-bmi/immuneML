import re


class StringHelper:

    @staticmethod
    def camel_case_to_words(camel_case_string: str):
        string = camel_case_string[0].upper() + camel_case_string[1:]
        return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)

    @staticmethod
    def camel_case_to_word_string(camel_case_string: str):
        return " ".join(StringHelper.camel_case_to_words(camel_case_string))
