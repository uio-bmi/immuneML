import os


class Util:

    @staticmethod
    def check_parameters(yaml_path, output_dir, kwargs, location):
        assert os.path.isfile(yaml_path), f"{location}: path to the specification is not correct, got {yaml_path}, " \
                                          f"expecting path to a YAML file."

        assert isinstance(output_dir, str) and output_dir != "", f"{location}: output_dir is {output_dir}, " \
                                                                 f"expected path to a folder to store the results."

        inputs = kwargs["inputs"].split(',') if "inputs" in kwargs else None
        assert "inputs" not in kwargs or all(os.path.dirname(inputs[0]) == os.path.dirname(elem) for elem in inputs), \
            f"{location}: not all repertoire files are under the same directory. " \
            f"Instead, they are in {str(list(os.path.dirname(elem) for elem in inputs))[1:-1]}."
