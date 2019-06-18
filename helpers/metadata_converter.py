import glob
import os
import sys

import pandas as pd


def convert_metadata(path, result_path, disease_name: str, negative_class_name: str, extension: str = "csv"):
    files = [f for f in glob.glob(path + "**/*.{}".format(extension), recursive=True)]

    donors = []
    chains = []
    disease = []
    filenames = []
    for file in files:
        filename = os.path.basename(file).split(".")[0]
        donors.append(filename.split('_')[0])
        chains.append(filename.split("_")[-1][-1])
        disease.append(True if disease_name in file and negative_class_name not in file else False)
        filenames.append(file)

    df = pd.DataFrame.from_dict({"donor": donors, "chain": chains, disease_name: disease, "filename": filenames})
    with open(result_path, "w") as file:
        df.to_csv(file, index=False, header=True)


if __name__ == "__main__":
    path = sys.argv[1]
    result_path = sys.argv[2]
    disease = sys.argv[3]
    alternative = sys.argv[4]
    extension = sys.argv[5] if len(sys.argv) == 6 else "csv"

    convert_metadata(path, result_path, disease, alternative, extension)
