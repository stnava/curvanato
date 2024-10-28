
# test_curvature.py
import numpy as np
import ants
import curvanato

def test_compute_curvature(radius):
    dim = (radius*2+5, radius*2+5, radius*2+5)
    center = (radius+2,radius+2,radius+2)
    spherical_volume = curvanato.create_spherical_volume(dim, radius, center)   
    spherical_volume_s = ants.smooth_image(spherical_volume, 0.5,sigma_in_physical_coordinates=True) 
    kvol = ants.weingarten_image_curvature( spherical_volume_s, 1.5 )
    # expected curvature is 1.0/r
    expected_k = 1.0 / float(radius)
    # evaluate the values at the surface
    spherical_volume_surf = spherical_volume - ants.iMath(spherical_volume,'ME',1)
    computed_k = kvol[ spherical_volume_surf == 1 ].mean()
    spherical_volume_surf2 = ants.iMath(spherical_volume,'MD',1) - spherical_volume
    computed_k2 = kvol[ spherical_volume_surf2 == 1 ].mean()
    computed_k_mean = 0.5 * computed_k2 + 0.5 * computed_k
    print( "computed : " + str(computed_k_mean))
    print( "expected_k : " + str(expected_k))
    print( computed_k_mean / expected_k )
    # assert result.shape == seg_image.shape
    # assert np.all(result == 0)

