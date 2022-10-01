from sensor_util.rotation_invariance import (
    project_onto_global_pc,
    project_onto_sliding_pc,
)

import numpy as np


def test_project_onto_global_or_sliding_pc():
    n = 30
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = np.random.rand(n)
    global_projection = project_onto_global_pc(x, y, z, num_pcs=2)
    sliding_projection = project_onto_sliding_pc(x, y, z, window_size=10, num_pcs=2)

    assert global_projection.shape[0] == n
    assert global_projection.shape[1] == 2
    assert sliding_projection.shape[0] == n
    assert sliding_projection.shape[1] == 2
