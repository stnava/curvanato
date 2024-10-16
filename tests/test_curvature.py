
# test_curvature.py
import numpy as np
from anato_curv import compute_curvature

def test_compute_curvature():
    seg_image = np.zeros((10, 10, 10))
    result = compute_curvature(seg_image)
    assert result.shape == seg_image.shape
    assert np.all(result == 0)

