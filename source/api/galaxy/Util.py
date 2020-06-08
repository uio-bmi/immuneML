import os


class Util:

    @staticmethod
    def check_parameters(yaml_path, output_dir, kwargs, location):
        assert os.path.isfile(yaml_path), f"{location}: path to the specification is not correct, got {yaml_path}, " \
                                          f"expecting path to a YAML file."

        assert isinstance(output_dir, str) and output_dir != "", f"{location}: output_dir is {output_dir}, " \
                                                                 f"expected path to a folder to store the results."
