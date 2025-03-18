import argparse
import glob
from pathlib import Path

from immuneML.app.ImmuneMLApp import ImmuneMLApp


def run_all_yamls(directory_path, result_path):

    yaml_files = list(glob.glob(f"{Path(directory_path)}/*.yaml"))

    if not yaml_files:
        print("No YAML files found in the specified directory.")
        return

    for yaml_file in yaml_files:
        yaml_path = Path(yaml_file)
        run_result_path = Path(result_path) / f"{yaml_path.stem}_result"
        print(f"\nProcessing {yaml_file}...")

        try:
            ImmuneMLApp(specification_path=yaml_path, result_path=run_result_path).run()
            print(f"Successfully completed {yaml_file}")
            print(f"Results saved to: {run_result_path}")
        except Exception as e:
            print(f"Error processing {yaml_file}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all YAML files in a directory using immuneML')
    parser.add_argument('--directory', '-d', default=".",
                        help='Directory containing YAML files (default: current directory)')
    parser.add_argument('--result-path', '-r', default="./results",
                        help='Directory to store results (default: ./results)')

    args = parser.parse_args()
    run_all_yamls(args.directory, args.result_path)
