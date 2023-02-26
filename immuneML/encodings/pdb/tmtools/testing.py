import pathlib


def get_pdb_path(sname):
    path = pathlib.Path(__file__).parent / "data" / f"{sname}.pdb"
    return path.resolve(strict=True)
