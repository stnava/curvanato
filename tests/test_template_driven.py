import ants
import curvanato
import sulceye
import numpy as np

def test_template_driven_partition():
    print("Running test_template_driven_partition...")
    # Create a small template sphere
    dim = (32, 32, 32)
    center = (16, 16, 16)
    template_img = curvanato.create_spherical_volume(dim, 10, center)
    
    # Create a slightly shifted subject sphere
    center_shift = (17, 16, 16)
    subject_img = curvanato.create_spherical_volume(dim, 10, center_shift)
    
    # Run subdivision on template
    template_subdiv = curvanato.subdivide_by_medial_axis(
        template_img, reference_axis=[1,0,0], prune_skeleton=False,
        smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
    )
    
    # Extract patches from template
    template_patches = sulceye.generate_patches_from_volume(template_subdiv)
    template_patch = template_patches[1]
    
    # Run template driven partition
    warped_patch, fwd, inv = curvanato.template_driven_partition(
        subject_image=subject_img,
        template_image=template_img,
        template_patch=template_patch,
        transform_type='Translation' # Fast registration
    )
    
    # Verify outputs
    assert warped_patch is not None, "Warped patch is None"
    assert len(warped_patch.vertices_3d) == len(template_patch.vertices_3d), "Vertex count mismatch"
    assert len(fwd) > 0, "Forward transforms list empty"
    assert len(inv) > 0, "Inverse transforms list empty"
    
    # Check that vertices were shifted by roughly (1, 0, 0) in absolute terms
    diff = np.mean(warped_patch.vertices_3d - template_patch.vertices_3d, axis=0)
    print("Mean coordinate shift:", diff)
    assert np.allclose(np.abs(diff), [1.0, 0.0, 0.0], atol=0.5), "Vertices not warped correctly"
    print("test_template_driven_partition passed successfully!")

if __name__ == "__main__":
    test_template_driven_partition()
