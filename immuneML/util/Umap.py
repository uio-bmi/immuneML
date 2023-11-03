from pathlib import Path

import numpy as np
import umap

from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


def run_umap(matrix: np.ndarray) -> np.ndarray:
    """
    Wrapper around the umap-learn-library.
    Runs umap on a matrix and returns the result
    as an n_examples x 2 matrix.
    """

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(matrix)
    return embedding
