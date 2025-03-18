import argparse
from pathlib import Path

from immuneML.data_model.bnp_util import read_yaml, write_yaml


def make_yamls(original_yaml_path: Path, result_yaml_path: Path):
    """
    Maps the section with MatchedSequencesEncoder in the yaml to multiple yaml files

    From this:
        definitions:
            encodings:
                my_ms_encoding:
                    MatchedSequences:
                        reference:
                            format: VDJDB
                            params:
                                path: [path/to/file.csv, path2.csv, path3.csv]
    To:

        file 1:

        definitions:
            encodings:
                my_ms_encoding:
                    MatchedSequences:
                        reference:
                            format: VDJDB
                            params:
                                path: path/to/file.csv

        another file:

        definitions:
            encodings:
                my_ms_encoding:
                    MatchedSequences:
                        reference:
                            format: VDJDB
                            params:
                                path: path2.csv


    Args:
        original_yaml_path: Path to the original file
        result_yaml_path: Path where the resulting yaml files will be stored

    Returns: None

    """
    original_yaml = read_yaml(original_yaml_path)

    assert "definitions" in original_yaml and "encodings" in original_yaml[
        'definitions'], f'Format not supported: {original_yaml}'
    assert all([list(enc.keys())[0] == "MatchedSequences" for enc in
                original_yaml['definitions']['encodings'].values()]), f"Encoding not supported: {original_yaml}"

    encoding_name = list(original_yaml['definitions']['encodings'].keys())[0]

    for i, path in enumerate(
            original_yaml['definitions']['encodings'][encoding_name]["MatchedSequences"]['reference']['params'][
                'path']):
        print(path)
        tmp_yaml = original_yaml.copy()
        tmp_yaml['definitions']['encodings'][encoding_name]["MatchedSequences"]['reference']['params']['path'] = path
        write_yaml(result_yaml_path / f'ref{i}.yaml', tmp_yaml)
        print(f"Created {result_yaml_path / f'ref{i}.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="immuneML yaml helper tool to expand ref sequence sets")
    parser.add_argument("original_path", help="Path to specification YAML file. Always used to define the "
                                              "analysis.")
    parser.add_argument("result_path", help="Output directory path.")

    namespace = parser.parse_args()
    make_yamls(Path(namespace.original_path), Path(namespace.result_path))


if __name__ == "__main__":
    main()
