# quality: gold

import errno
import os


class PathBuilder:

    @staticmethod
    def build(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        return path
