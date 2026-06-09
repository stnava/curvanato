import ants
import numpy as np
import curvanato
import sulceye
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import siq

def main():
    print("Loading Baseline Caudate...")
    caud_vent = ants.image_read("curvanato/data/caud_vent.nii.gz")
    base_caud = ants.threshold_image(caud_vent, 50, 50)
    base_caud = ants.crop_image(base_caud, ants.iMath(base_caud, "MD", 10))
    
    base_caud_hr = siq.auto(base_caud)
    base_caud_hr = ants.threshold_image(base_caud_hr, 0.5, 1.5).threshold_image(1, 1)
    base_caud_hr = ants.iMath(base_caud_hr, "GetLargestComponent")
    
    subdiv = curvanato.subdivide_by_medial_axis(base_caud_hr, reference_axis=[1,0,0], prune_skeleton=True, smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5)
    dist = ants.iMath(base_caud_hr, "MaurerDistance")
    
    fig = plt.figure(figsize=(18, 9))
    
    plot_idx = 1
    
    for group in ['Control', 'Disease']:
        for i in range(4):
            np.random.seed(hash(f"{group}_{i}") % (2**32))
            variability = np.random.uniform(-0.5, 0.5)
            
            dist_map = dist.clone()
            if group == 'Disease':
                bias = np.zeros_like(base_caud_hr.numpy())
                medial_mask = (subdiv.numpy() == 1)
                y_coords = np.indices(base_caud_hr.shape)[1]
                lesion_mask = medial_mask & (y_coords > 73)
                bias[lesion_mask] = -1.5
                dist_map = dist_map + base_caud_hr.new_image_like(bias)
                
            dist_mod = dist_map + variability
            img_subj = ants.threshold_image(dist_mod, -100, -0.5)
            img_subj = ants.iMath(img_subj, "GetLargestComponent")
            
            # Fast medial subdivision for subject
            subdiv_subj = curvanato.subdivide_by_medial_axis(img_subj, reference_axis=[1,0,0], prune_skeleton=True, smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5)
            
            patches = sulceye.generate_patches_from_volume(subdiv_subj)
            p = patches[1] # Medial region
            
            ax = fig.add_subplot(2, 4, plot_idx, projection='3d')
            # Extract thickness roughly for coloring
            curv_subj = ants.weingarten_image_curvature(ants.smooth_image(ants.iMath(img_subj, "MaurerDistance"), 1.5), 1.5, 0)
            
            ax.plot_trisurf(p.vertices_3d[:,0], p.vertices_3d[:,1], p.vertices_3d[:,2], 
                            triangles=p.faces_local, cmap='viridis', edgecolor='none', alpha=0.9)
            
            ax.set_title(f"{group} Subject {i+1}")
            ax.view_init(elev=20, azim=45)
            ax.axis('off')
            plot_idx += 1
            
    plt.tight_layout()
    fig_path = '/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/surfaces_grid.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved {fig_path}")

if __name__ == "__main__":
    main()
