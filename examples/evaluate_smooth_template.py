import ants
import numpy as np
import time
import pandas as pd
import curvanato
import sulceye
import scipy
import scipy.special
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import skimage.measure
from scipy.spatial import cKDTree
from scipy import stats
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import warnings
import os

warnings.filterwarnings('ignore', category=DeprecationWarning)

def compute_mesh_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v2 - v0, v1 - v0)
    for j in range(3):
        np.add.at(normals, faces[:, j], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    return normals

def spectral_mesh_smoothing(vertices, faces, k=35):
    N = len(vertices)
    rows = []
    cols = []
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    rows.append(f[i])
                    cols.append(f[j])
                      
    data = np.ones(len(rows), dtype=float)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.data = np.ones_like(A.data)
    
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    L_un = sp.diags(degrees) - A
    L_norm = D_inv_sqrt @ L_un @ D_inv_sqrt
    
    # Use shift-invert solver for stability
    eigenvalues, U = eigsh(L_norm, k=k, sigma=1e-6, which='LM')
    
    d_sqrt = 1.0 / d_inv_sqrt
    D_sqrt = sp.diags(d_sqrt)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Project: V_smooth = D^{-1/2} U U^T D^{1/2} V
    vertices_smooth = D_inv_sqrt @ (U @ (U.T @ (D_sqrt @ vertices)))
    return vertices_smooth

def voxelize_mesh(vertices_phys, faces, reference_image):
    normals_phys = compute_mesh_normals(vertices_phys, faces)
    shape = reference_image.shape
    spacing = np.array(reference_image.spacing)
    origin = np.array(reference_image.origin)
    direction = np.array(reference_image.direction)
    
    grid_indices = np.argwhere(np.ones(shape))
    grid_phys = (grid_indices * spacing) @ direction.T + origin
    
    tree = cKDTree(vertices_phys)
    dists, indices = tree.query(grid_phys)
    
    dirs = grid_phys - vertices_phys[indices]
    dot_prods = np.sum(dirs * normals_phys[indices], axis=1)
    
    # Require dot product <= 0 AND distance to nearest vertex is small (e.g., < 4.0 mm)
    inside = (dot_prods <= 0.0) & (dists < 4.0)
    
    binary_np = np.zeros(shape, dtype=np.float32)
    binary_np[tuple(grid_indices[inside].T)] = 1.0
    
    binary_img = ants.from_numpy(binary_np)
    binary_img = ants.copy_image_info(reference_image, binary_img)
    return binary_img

def sample_scalars(p, img3d, inset_mm=1.0):
    inv_dir = np.linalg.inv(img3d.direction)
    origin = np.array(img3d.origin)
    spacing = np.array(img3d.spacing)
    inset_vertices = p.vertices_3d - (p.normals_3d * inset_mm)
    
    indices = ((inset_vertices - origin) @ inv_dir.T) / spacing
    indices = np.round(indices).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, img3d.shape[0]-1)
    indices[:, 1] = np.clip(indices[:, 1], 0, img3d.shape[1]-1)
    indices[:, 2] = np.clip(indices[:, 2], 0, img3d.shape[2]-1)
    
    return img3d.numpy()[indices[:, 0], indices[:, 1], indices[:, 2]]

def compute_thickness_raycast(p, img3d, max_dist=15.0, step=0.2):
    inv_dir = np.linalg.inv(img3d.direction)
    origin = np.array(img3d.origin)
    spacing = np.array(img3d.spacing)
    
    thickness = np.zeros(len(p.vertices_3d))
    img_data = img3d.numpy()
    
    for d in np.arange(step, max_dist, step):
        test_vertices = p.vertices_3d - (p.normals_3d * d)
        indices = ((test_vertices - origin) @ inv_dir.T) / spacing
        indices = np.round(indices).astype(int)
        indices[:, 0] = np.clip(indices[:, 0], 0, img3d.shape[0]-1)
        indices[:, 1] = np.clip(indices[:, 1], 0, img3d.shape[1]-1)
        indices[:, 2] = np.clip(indices[:, 2], 0, img3d.shape[2]-1)
        
        vals = img_data[indices[:, 0], indices[:, 1], indices[:, 2]]
        hit_bg = (vals == 0)
        update_mask = hit_bg & (thickness == 0)
        thickness[update_mask] = d
        
    thickness[thickness == 0] = max_dist
    return thickness

def compute_boundary_area(subdiv):
    r1 = ants.threshold_image(subdiv, 1, 1)
    r2 = ants.threshold_image(subdiv, 2, 2)
    r1_dilated = ants.iMath(r1, "MD", 1)
    boundary = r1_dilated * r2
    return boundary.sum()

def main():
    print("Loading Baseline Caudate...")
    caud_vent = ants.image_read("curvanato/data/caud_vent.nii.gz")
    base_caud = ants.threshold_image(caud_vent, 50, 50)
    base_caud = ants.crop_image(base_caud, ants.iMath(base_caud, "MD", 10))
    
    print("Skipping Super-Resolution, using cropped base caudate template directly...")
    base_caud_hr = base_caud
    
    # Get physical boundary mesh of template to smooth it
    print("Extracting Template boundary and smoothing...")
    img_np = base_caud_hr.numpy()
    verts_idx, faces_temp, _, _ = skimage.measure.marching_cubes(img_np, level=0.5)
    spacing_temp = np.array(base_caud_hr.spacing)
    origin_temp = np.array(base_caud_hr.origin)
    dir_temp = np.array(base_caud_hr.direction)
    verts_temp_phys = (verts_idx * spacing_temp) @ dir_temp.T + origin_temp
    
    # Smooth the template boundary mesh
    verts_temp_smooth = spectral_mesh_smoothing(verts_temp_phys, faces_temp, k=35)
    
    # Voxelize smoothed template
    print("Voxelizing smooth template...")
    base_caud_hr_smooth = voxelize_mesh(verts_temp_smooth, faces_temp, base_caud_hr)
    base_caud_hr_smooth = ants.iMath(base_caud_hr_smooth, "GetLargestComponent")
    
    print("Generating template subdiv & baseline distance map...")
    dist = ants.iMath(base_caud_hr_smooth, "MaurerDistance")
    
    # Pre-calculate baseline medial axis on the smoothed template
    template_subdiv_smooth = curvanato.subdivide_by_medial_axis(
        base_caud_hr_smooth, reference_axis=[1,0,0], prune_skeleton=True,
        smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
    )
    template_patches = sulceye.generate_patches_from_volume(template_subdiv_smooth, method='spectral')
    template_patch_smooth = template_patches[1]
    
    # Storage for results
    grids_curv = {'Control': [], 'Disease': []}
    grids_thick = {'Control': [], 'Disease': []}
    timing_data = []
    consistency_data = []
    
    # Keep track of first 3 subject shapes & warped patches for visualization
    subj_visual_data = []
    
    n_subj = 10
    print(f"Simulating {n_subj} subjects per group...")
    
    # Create figs folder if it doesn't exist
    os.makedirs('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs', exist_ok=True)
    
    for group in ['Control', 'Disease']:
        for i in range(n_subj):
            print(f"\n--- Processing {group} Subject {i+1}/{n_subj} ---")
            np.random.seed(hash(f"{group}_{i}") % (2**32))
            variability = np.random.uniform(-0.5, 0.5)
            
            # Note: The subject is simulated from base_caud_hr (the original noisy shape)
            # to show how our smooth template propagates to raw noisy target subjects!
            dist_map = ants.iMath(base_caud_hr, "MaurerDistance")
            if group == 'Disease':
                bias = np.zeros_like(base_caud_hr.numpy())
                medial_mask = (template_subdiv_smooth.numpy() == 1)
                y_coords = np.indices(base_caud_hr.shape)[1]
                lesion_mask = medial_mask & (y_coords > 36)
                bias[lesion_mask] = -1.5
                dist_map = dist_map + base_caud_hr.new_image_like(bias)
                
            dist_mod = dist_map + variability
            img_subj = ants.threshold_image(dist_mod, -100, -0.5)
            img_subj = ants.iMath(img_subj, "GetLargestComponent")
            
            # Subject Curvature metric image
            dist_subj = ants.iMath(img_subj, "MaurerDistance")
            smooth_dist_subj = ants.smooth_image(dist_subj, 1.5)
            curv_subj = ants.weingarten_image_curvature(smooth_dist_subj, 1.5, 0)
            
            # Smooth Template-Driven Registration Method
            t0 = time.time()
            try:
                # Register smooth template to raw subject image
                warped_patch, fwd, inv = curvanato.template_driven_partition(
                    subject_image=img_subj,
                    template_image=base_caud_hr_smooth,
                    template_patch=template_patch_smooth,
                    transform_type='SyNOnly'
                )
                
                # Sample Curvature
                curv_val = sample_scalars(warped_patch, curv_subj, inset_mm=1.0)
                warped_patch.scalars = curv_val
                grid_c = warped_patch.to_grid(resolution=(100, 50))
                grids_curv[group].append(ndimage.gaussian_filter(grid_c, sigma=1.0))
                
                # Sample Thickness
                thick_val = compute_thickness_raycast(warped_patch, img_subj)
                warped_patch.scalars = thick_val
                grid_t = warped_patch.to_grid(resolution=(100, 50))
                grids_thick[group].append(ndimage.gaussian_filter(grid_t, sigma=1.0))
                
                # Warp template subdiv to subject space to calculate boundary area
                subdiv_temp_warped = ants.apply_transforms(
                    fixed=img_subj, moving=template_subdiv_smooth,
                    transformlist=fwd, interpolator='nearestNeighbor'
                )
                
                timing_data.append(time.time() - t0)
                consistency_data.append(compute_boundary_area(subdiv_temp_warped))
                
                # Save first 3 subjects for rendering
                if group == 'Control' and len(subj_visual_data) < 3:
                    # Extract subject's original boundary mesh (unregularized)
                    subj_np = img_subj.numpy()
                    s_verts_idx, s_faces, _, _ = skimage.measure.marching_cubes(subj_np, level=0.5)
                    s_spacing = np.array(img_subj.spacing)
                    s_origin = np.array(img_subj.origin)
                    s_dir = np.array(img_subj.direction)
                    s_verts_phys = (s_verts_idx * s_spacing) @ s_dir.T + s_origin
                    
                    subj_visual_data.append({
                        'verts_bound': s_verts_phys,
                        'faces_bound': s_faces,
                        'verts_patch': warped_patch.vertices_3d.copy(),
                        'faces_patch': warped_patch.faces_local.copy() if warped_patch.faces_local is not None else None
                    })
            except Exception as e:
                print("Registration failed for this subject:", e)
                
    # --- Statistical Analysis ---
    print("\n--- Running Group Comparisons ---")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    metrics_info = [("Curvature", grids_curv), ("Thickness", grids_thick)]
    
    stat_summary = []
    
    for row, (metric_name, grids_dict) in enumerate(metrics_info):
        ax = axes[row]
        ctrl = np.stack(grids_dict['Control'])
        dis = np.stack(grids_dict['Disease'])
        
        t_stat, p_val = stats.ttest_ind(dis, ctrl, axis=0)
        
        p_flat = p_val.flatten()
        valid_mask = ~np.isnan(p_flat)
        reject = np.zeros_like(p_flat, dtype=bool)
        if len(p_flat[valid_mask]) > 0:
            reject_valid, _, _, _ = multipletests(p_flat[valid_mask], alpha=0.05, method='holm')
            reject[valid_mask] = reject_valid
        sig_mask = reject.reshape(p_val.shape)
        
        t_plot = t_stat.copy()
        t_plot[~sig_mask] = np.nan
        
        im = ax.imshow(t_plot, cmap='coolwarm', vmin=-6, vmax=6)
        ax.set_title(f"Smooth Template-Driven: {metric_name} Difference (FWE p<0.05)", fontsize=14, fontweight='bold')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        
        max_t = np.nanmax(np.abs(t_stat))
        sig_pixels = np.sum(sig_mask)
        stat_summary.append({
            'Metric': metric_name,
            'Max |T|': max_t,
            'Significant Pixels': sig_pixels
        })
        
    plt.tight_layout()
    stats_fig_path = '/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/smooth_template_comparison_stats.png'
    plt.savefig(stats_fig_path, dpi=150)
    print(f"Saved stats figure to {stats_fig_path}")
    
    # Save performance & consistency stats
    df_perf = pd.DataFrame([{
        'Mean Speed (s)': np.mean(timing_data),
        'Std Speed (s)': np.std(timing_data),
        'Mean Boundary Area': np.mean(consistency_data),
        'Std Boundary Area': np.std(consistency_data),
    }])
    
    df_stats = pd.DataFrame(stat_summary)
    
    df_perf.to_csv('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/smooth_performance_results.csv', index=False)
    df_stats.to_csv('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/smooth_statistical_results.csv', index=False)
    
    # Save the raw 2D grids for unified plotting
    np.save('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/grids_curv_smooth.npy', grids_curv)
    np.save('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/grids_thick_smooth.npy', grids_thick)
    
    print("\n--- Performance Summary ---")
    print(df_perf.to_markdown(index=False))
    print("\n--- Statistical Power Summary ---")
    print(df_stats.to_markdown(index=False))
    
    # --- Generate 3D Overlays Grid Figure ---
    print("\n--- Generating 3D Shape Overlays Grid ---")
    import matplotlib.patches as mpatches
    fig3d = plt.figure(figsize=(20, 16), facecolor='white')
    
    # 1. Plot Template Shape (Original noisy base_caud_hr in light gray, and smooth patch in indigo)
    ax1 = fig3d.add_subplot(221, projection='3d')
    ax1.set_facecolor('white')
    
    # Extract original noisy template boundary vertices for display
    noisy_temp_np = base_caud_hr.numpy()
    nt_verts_idx, nt_faces, _, _ = skimage.measure.marching_cubes(noisy_temp_np, level=0.5)
    nt_verts_phys = (nt_verts_idx * spacing_temp) @ dir_temp.T + origin_temp
    
    # Plot original boundary using plot_trisurf
    ax1.plot_trisurf(nt_verts_phys[:, 0], nt_verts_phys[:, 1], nt_verts_phys[:, 2], 
                     triangles=nt_faces, color='#cbd5e1', alpha=0.15, edgecolor='#94a3b8', linewidth=0.05)
    
    # Plot smooth medial axis using plot_trisurf (since it's a surface patch)
    if template_patch_smooth.faces_local is not None and len(template_patch_smooth.faces_local) > 0:
        ax1.plot_trisurf(template_patch_smooth.vertices_3d[:, 0], template_patch_smooth.vertices_3d[:, 1], template_patch_smooth.vertices_3d[:, 2], 
                         triangles=template_patch_smooth.faces_local, color='#4f46e5', alpha=0.75, edgecolor='#312e81', linewidth=0.1)
    else:
        ax1.scatter(template_patch_smooth.vertices_3d[:, 0], template_patch_smooth.vertices_3d[:, 1], template_patch_smooth.vertices_3d[:, 2], 
                   color='#4f46e5', s=8, alpha=0.85)
                   
    # Legend
    temp_boundary_patch = mpatches.Patch(color='#cbd5e1', alpha=0.3, label='Original Template Boundary')
    temp_ma_patch = mpatches.Patch(color='#4f46e5', alpha=0.8, label='Smooth Medial Axis')
    ax1.legend(handles=[temp_boundary_patch, temp_ma_patch], loc='upper right')
    ax1.set_title("Template: Original Noisy boundary\n& Smooth Medial Axis", fontsize=15, fontweight='bold')
    
    # Plot target subjects
    for idx, subj in enumerate(subj_visual_data):
        ax = fig3d.add_subplot(222 + idx, projection='3d')
        ax.set_facecolor('white')
        
        # Plot subject boundary using plot_trisurf
        ax.plot_trisurf(subj['verts_bound'][:, 0], subj['verts_bound'][:, 1], subj['verts_bound'][:, 2], 
                        triangles=subj['faces_bound'], color='#99f6e4', alpha=0.15, edgecolor='#2dd4bf', linewidth=0.05)
        
        # Plot warped medial axis using plot_trisurf
        if subj['faces_patch'] is not None and len(subj['faces_patch']) > 0:
            ax.plot_trisurf(subj['verts_patch'][:, 0], subj['verts_patch'][:, 1], subj['verts_patch'][:, 2], 
                            triangles=subj['faces_patch'], color='#f43f5e', alpha=0.75, edgecolor='#9f1239', linewidth=0.1)
        else:
            ax.scatter(subj['verts_patch'][:, 0], subj['verts_patch'][:, 1], subj['verts_patch'][:, 2], 
                       color='#f43f5e', s=8, alpha=0.85)
                       
        subj_boundary_patch = mpatches.Patch(color='#99f6e4', alpha=0.3, label=f'Subject {idx+1} Original Boundary')
        subj_ma_patch = mpatches.Patch(color='#f43f5e', alpha=0.8, label='Warped Medial Axis')
        ax.legend(handles=[subj_boundary_patch, subj_ma_patch], loc='upper right')
        ax.set_title(f"Target Subject {idx+1}:\nOriginal Shape & Warped Medial Axis", fontsize=15, fontweight='bold')
        
    # Standardize view angles and ranges
    all_axes = [ax1, fig3d.axes[1], fig3d.axes[2], fig3d.axes[3]]
    for idx, ax in enumerate(all_axes):
        # We can center the views based on their coordinates
        if idx == 0:
            coords = nt_verts_phys
        else:
            coords = subj_visual_data[idx-1]['verts_bound']
            
        min_c = np.min(coords, axis=0)
        max_c = np.max(coords, axis=0)
        center = (min_c + max_c) / 2.0
        max_r = np.max(max_c - min_c)
        ax.set_xlim(center[0] - max_r/2, center[0] + max_r/2)
        ax.set_ylim(center[1] - max_r/2, center[1] + max_r/2)
        ax.set_zlim(center[2] - max_r/2, center[2] + max_r/2)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
    plt.tight_layout()
    grid_fig_path = '/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/smooth_template_mapping_grid.png'
    plt.savefig(grid_fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D overlay grid figure to {grid_fig_path}")

if __name__ == "__main__":
    main()
