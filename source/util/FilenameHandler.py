import re


class FilenameHandler:

    @staticmethod
    def _to_snake_case(string: str):
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)).lower()

    @staticmethod
    def get_filename(class_name: str, file_type: str):
        """
        converts the class name to snake case and appends given file type
        :param class_name: name of the class that will be stored in the file
        :param file_type: file extension: pickle, json
        :return: filename consisting of concatenated class_name in snake case and file type
        """
        name = FilenameHandler._to_snake_case(class_name)
        if file_type != "":
            name += ".{}".format(file_type)

        return name

    @staticmethod
    def get_dataset_name(class_name: str):
        if "Encoder" in class_name:
            name = "encoded_dataset.iml_dataset"
        else:
            name = "dataset.iml_dataset"

        return name

    @staticmethod
    def get_model_name(class_name: str, file_type: str = "pickle"):
        name = FilenameHandler._to_snake_case(class_name) + "_model." + file_type
        return name
