# quality: gold

import errno
import os
import warnings


class PathBuilder:

    @staticmethod
    def build(path, warn_if_exists=False):
        if warn_if_exists and os.path.isdir(path):
            warnings.warn(f"PathBuilder: directory {path} already exists. Writing in the existing directory...", RuntimeWarning)
        else:
            try:
                path = path + "/" if path[-1] != "/" else path
                os.makedirs(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        return path
