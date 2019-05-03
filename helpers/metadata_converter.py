import glob
import os
import sys

import pandas as pd


def convert_metadata(path, result_path, extension: str = "csv"):
    files = [f for f in glob.glob(path + "**/*.{}".format(extension), recursive=True)]

    donors = []
    chains = []
    celiacs = []
    filenames = []
    for file in files:
        filename = os.path.basename(file).split(".")[0]
        donors.append(filename.split('_')[0])
        chains.append(filename.split("_")[2][-1])
        celiacs.append(True if "_CD" in file and "_HC" not in file else False)
        filenames.append(file)

    df = pd.DataFrame.from_dict({"donor": donors, "chain": chains, "celiac": celiacs, "filename": filenames})
    with open(result_path, "w") as file:
        df.to_csv(file, sep="\t", index=False, header=True)


if __name__ == "__main__":
    path = sys.argv[1]
    result_path = sys.argv[2]
    extension = sys.argv[3] if len(sys.argv) == 4 else "csv"

    convert_metadata(path, result_path, extension)
