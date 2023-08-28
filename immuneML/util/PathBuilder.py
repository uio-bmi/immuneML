# quality: gold

import errno
import os
import shutil
import warnings
from pathlib import Path


class PathBuilder:

    @staticmethod
    def build(path, warn_if_exists=False):
        path = Path(path)
        if warn_if_exists and path.is_dir():
            warnings.warn(f"PathBuilder: directory {path} already exists. Writing in the existing directory...", RuntimeWarning)
        else:
            try:
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        return path

    @staticmethod
    def remove_old_and_build(path):
        path = Path(path)
        if path.is_dir():
            shutil.rmtree(path)

        path.mkdir(parents=True)

        return path
