import numpy as np
from immuneML.util.Umap import run_umap


# simple acceptance test
def test_umap_wrapper():
    data = np.random.randint(0, 10, (4, 4))
    result = run_umap(data)

