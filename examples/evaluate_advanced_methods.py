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
import copy

# Suppress deprecation warnings from scipy sph_harm
warnings.filterwarnings('ignore', category=DeprecationWarning)

def smooth_image_with_nans(image, sigma=1.0):
    nan_mask = np.isnan(image)
    im = image.copy()
    im[nan_mask] = 0.0
    w = (~nan_mask).astype(float)
    
    im_smooth = ndimage.gaussian_filter(im, sigma=sigma)
    w_smooth = ndimage.gaussian_filter(w, sigma=sigma)
    
    w_smooth[w_smooth == 0] = 1e-8
    result = im_smooth / w_smooth
    result[nan_mask] = np.nan
    return result

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

def parameterize_to_sphere(vertices, faces, n_iter=50):
    from collections import defaultdict
    adj = defaultdict(set)
    for f in faces:
        adj[f[0]].add(f[1])
        adj[f[0]].add(f[2])
        adj[f[1]].add(f[0])
        adj[f[1]].add(f[2])
        adj[f[2]].add(f[0])
        adj[f[2]].add(f[1])
        
    N = len(vertices)
    V_c = vertices - np.mean(vertices, axis=0)
    S = V_c / np.linalg.norm(V_c, axis=1, keepdims=True)
    
    for _ in range(n_iter):
        S_new = np.zeros_like(S)
        for i in range(N):
            neighbors = list(adj[i])
            if len(neighbors) > 0:
                S_new[i] = np.mean(S[neighbors], axis=0)
            else:
                S_new[i] = S[i]
        norms = np.linalg.norm(S_new, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        S = S_new / norms
    return S

def get_spharm_basis(theta, phi, L_max):
    basis = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            y = scipy.special.sph_harm(m, l, phi, theta)
            if m > 0:
                y_real = np.sqrt(2) * np.real(y)
            elif m < 0:
                y_real = np.sqrt(2) * np.imag(scipy.special.sph_harm(-m, l, phi, theta))
            else:
                y_real = np.real(y)
            basis.append(y_real)
    return np.column_stack(basis)

def fit_spharm(vertices, S, L_max):
    theta = np.arccos(np.clip(S[:, 2], -1.0, 1.0))
    phi = np.arctan2(S[:, 1], S[:, 0])
    B = get_spharm_basis(theta, phi, L_max)
    coeff_x, _, _, _ = np.linalg.lstsq(B, vertices[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(B, vertices[:, 1], rcond=None)
    coeff_z, _, _, _ = np.linalg.lstsq(B, vertices[:, 2], rcond=None)
    vertices_smooth = np.column_stack([
        B @ coeff_x,
        B @ coeff_y,
        B @ coeff_z
    ])
    return vertices_smooth

def spectral_mesh_smoothing(vertices, faces, k=40):
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
    
    base_caud_hr = base_caud
    dist = ants.iMath(base_caud_hr, "MaurerDistance")
    
    # Pre-calculate baseline medial axis for template-driven registration
    print("Generating template subdiv & baseline distance map...")
    template_subdiv = curvanato.subdivide_by_medial_axis(
        base_caud_hr, reference_axis=[1,0,0], prune_skeleton=True,
        smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
    )
    
    # Generate Ribbon (prior), Spectral (new) and ARAP template patches
    print("Extracting template patches...")
    template_patches_ribbon = sulceye.generate_patches_from_volume(template_subdiv, method='ribbon')
    template_patch_ribbon = template_patches_ribbon[1]
    
    template_patches_spec = sulceye.generate_patches_from_volume(template_subdiv, method='spectral')
    template_patch_spec = template_patches_spec[1]
    
    template_patches_arap = sulceye.generate_patches_from_volume(template_subdiv, method='arap')
    template_patch_arap = template_patches_arap[1]
    
    # Generate Smooth Template
    print("Extracting Template boundary and smoothing...")
    img_np_temp = base_caud_hr.numpy()
    verts_idx_temp, faces_temp, _, _ = skimage.measure.marching_cubes(img_np_temp, level=0.5)
    spacing_temp = np.array(base_caud_hr.spacing)
    origin_temp = np.array(base_caud_hr.origin)
    dir_temp = np.array(base_caud_hr.direction)
    verts_temp_phys = (verts_idx_temp * spacing_temp) @ dir_temp.T + origin_temp
    
    # Smooth the template boundary mesh and voxelize it
    verts_temp_smooth = spectral_mesh_smoothing(verts_temp_phys, faces_temp, k=35)
    base_caud_hr_smooth = voxelize_mesh(verts_temp_smooth, faces_temp, base_caud_hr)
    base_caud_hr_smooth = ants.iMath(base_caud_hr_smooth, "GetLargestComponent")
    
    template_subdiv_smooth = curvanato.subdivide_by_medial_axis(
        base_caud_hr_smooth, reference_axis=[1,0,0], prune_skeleton=True,
        smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
    )
    # Smooth Template uses Spectral parameterization coordinates
    template_patches_smooth = sulceye.generate_patches_from_volume(template_subdiv_smooth, method='spectral')
    template_patch_smooth = template_patches_smooth[1]
    
    # Pre-construct points DataFrame from canonical template vertices (for warping)
    pts_df_raw = pd.DataFrame({
        'x': template_patch_ribbon.vertices_3d[:, 0],
        'y': template_patch_ribbon.vertices_3d[:, 1],
        'z': template_patch_ribbon.vertices_3d[:, 2]
    })
    
    pts_df_smooth = pd.DataFrame({
        'x': template_patch_smooth.vertices_3d[:, 0],
        'y': template_patch_smooth.vertices_3d[:, 1],
        'z': template_patch_smooth.vertices_3d[:, 2]
    })
    
    methods = ['Baseline', 'SPHARM', 'Spectral', 'Template (Ribbon)', 'Template (Spectral)', 'Smooth Template', 'Template (ARAP)']
    
    # Setup results storage
    grids_curv = {m: {'Control': [], 'Disease': []} for m in methods}
    grids_thick = {m: {'Control': [], 'Disease': []} for m in methods}
    timing_data = {m: [] for m in methods}
    consistency_data = {m: [] for m in methods}
    
    n_subj = 10
    print(f"Simulating {n_subj} subjects per group...")
    
    os.makedirs('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs', exist_ok=True)
    
    for group in ['Control', 'Disease']:
        for i in range(n_subj):
            print(f"\n--- Processing {group} Subject {i+1}/{n_subj} ---")
            np.random.seed(hash(f"{group}_{i}") % (2**32))
            variability = np.random.uniform(-0.5, 0.5)
            
            dist_map = dist.clone()
            if group == 'Disease':
                bias = np.zeros_like(base_caud_hr.numpy())
                medial_mask = (template_subdiv.numpy() == 1)
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
            
            # 1. Baseline Method (Uses Ribbon parameterization)
            t0 = time.time()
            subdiv_base = curvanato.subdivide_by_medial_axis(
                img_subj, reference_axis=[1,0,0], prune_skeleton=True,
                smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
            )
            try:
                patches_base = sulceye.generate_patches_from_volume(subdiv_base, method='ribbon')
                p_base = patches_base[1]
                
                curv_val = sample_scalars(p_base, curv_subj, inset_mm=1.0)
                curv_val = curv_val + np.random.normal(0, 0.02, size=curv_val.shape)
                p_base.scalars = curv_val
                grid_c = p_base.to_grid(resolution=(100, 50))
                grids_curv['Baseline'][group].append(smooth_image_with_nans(grid_c, sigma=1.0))
                
                thick_val = compute_thickness_raycast(p_base, img_subj)
                thick_val = thick_val + np.random.normal(0, 0.1, size=thick_val.shape)
                p_base.scalars = thick_val
                grid_t = p_base.to_grid(resolution=(100, 50))
                grids_thick['Baseline'][group].append(smooth_image_with_nans(grid_t, sigma=1.0))
                
                timing_data['Baseline'].append(time.time() - t0)
                consistency_data['Baseline'].append(compute_boundary_area(subdiv_base))
            except Exception as e:
                print("Baseline failed for this subject:", e)
                
            # Warp coordinates for Template-driven methods
            t0 = time.time()
            try:
                reg_raw = ants.registration(fixed=img_subj, moving=base_caud_hr, type_of_transform='SyNOnly')
                pts_warped_df_raw = ants.apply_transforms_to_points(3, pts_df_raw, reg_raw['invtransforms'])
                pts_warped_raw = pts_warped_df_raw[['x', 'y', 'z']].values
                dt_reg_raw = time.time() - t0
                
                # 2. Template (Ribbon)
                p_temp_ribbon = copy.copy(template_patch_ribbon)
                p_temp_ribbon.vertices_3d = pts_warped_raw
                p_temp_ribbon.normals_3d = compute_mesh_normals(pts_warped_raw, template_patch_ribbon.faces_local)
                
                curv_val = sample_scalars(p_temp_ribbon, curv_subj, inset_mm=1.0)
                curv_val = curv_val + np.random.normal(0, 0.02, size=curv_val.shape)
                p_temp_ribbon.scalars = curv_val
                grid_c = p_temp_ribbon.to_grid(resolution=(100, 50))
                grids_curv['Template (Ribbon)'][group].append(smooth_image_with_nans(grid_c, sigma=1.0))
                
                thick_val = compute_thickness_raycast(p_temp_ribbon, img_subj)
                thick_val = thick_val + np.random.normal(0, 0.1, size=thick_val.shape)
                p_temp_ribbon.scalars = thick_val
                grid_t = p_temp_ribbon.to_grid(resolution=(100, 50))
                grids_thick['Template (Ribbon)'][group].append(smooth_image_with_nans(grid_t, sigma=1.0))
                
                # 3. Template (Spectral)
                p_temp_spec = copy.copy(template_patch_spec)
                p_temp_spec.vertices_3d = pts_warped_raw
                p_temp_spec.normals_3d = p_temp_ribbon.normals_3d
                
                curv_val_spec = sample_scalars(p_temp_spec, curv_subj, inset_mm=1.0)
                curv_val_spec = curv_val_spec + np.random.normal(0, 0.02, size=curv_val_spec.shape)
                p_temp_spec.scalars = curv_val_spec
                grid_c_spec = p_temp_spec.to_grid(resolution=(100, 50))
                grids_curv['Template (Spectral)'][group].append(smooth_image_with_nans(grid_c_spec, sigma=1.0))
                
                thick_val_spec = compute_thickness_raycast(p_temp_spec, img_subj)
                thick_val_spec = thick_val_spec + np.random.normal(0, 0.1, size=thick_val_spec.shape)
                p_temp_spec.scalars = thick_val_spec
                grid_t_spec = p_temp_spec.to_grid(resolution=(100, 50))
                grids_thick['Template (Spectral)'][group].append(smooth_image_with_nans(grid_t_spec, sigma=1.0))
                
                # Template (ARAP)
                p_temp_arap = copy.copy(template_patch_arap)
                p_temp_arap.vertices_3d = pts_warped_raw
                p_temp_arap.normals_3d = p_temp_ribbon.normals_3d
                
                curv_val_arap = sample_scalars(p_temp_arap, curv_subj, inset_mm=1.0)
                curv_val_arap = curv_val_arap + np.random.normal(0, 0.02, size=curv_val_arap.shape)
                p_temp_arap.scalars = curv_val_arap
                grid_c_arap = p_temp_arap.to_grid(resolution=(100, 50))
                grids_curv['Template (ARAP)'][group].append(smooth_image_with_nans(grid_c_arap, sigma=1.0))
                
                thick_val_arap = compute_thickness_raycast(p_temp_arap, img_subj)
                thick_val_arap = thick_val_arap + np.random.normal(0, 0.1, size=thick_val_arap.shape)
                p_temp_arap.scalars = thick_val_arap
                grid_t_arap = p_temp_arap.to_grid(resolution=(100, 50))
                grids_thick['Template (ARAP)'][group].append(smooth_image_with_nans(grid_t_arap, sigma=1.0))
                
                subdiv_temp_warped = ants.apply_transforms(
                    fixed=img_subj, moving=template_subdiv,
                    transformlist=reg_raw['fwdtransforms'], interpolator='nearestNeighbor'
                )
                
                timing_data['Template (Ribbon)'].append(dt_reg_raw)
                timing_data['Template (Spectral)'].append(dt_reg_raw)
                timing_data['Template (ARAP)'].append(dt_reg_raw)
                area_temp = compute_boundary_area(subdiv_temp_warped)
                consistency_data['Template (Ribbon)'].append(area_temp)
                consistency_data['Template (Spectral)'].append(area_temp)
                consistency_data['Template (ARAP)'].append(area_temp)
            except Exception as e:
                print("Template registration failed for this subject:", e)
                
            # 4. Smooth Template Method
            t0 = time.time()
            try:
                reg_smooth = ants.registration(fixed=img_subj, moving=base_caud_hr_smooth, type_of_transform='SyNOnly')
                pts_warped_df_smooth = ants.apply_transforms_to_points(3, pts_df_smooth, reg_smooth['invtransforms'])
                pts_warped_smooth = pts_warped_df_smooth[['x', 'y', 'z']].values
                dt_reg_smooth = time.time() - t0
                
                p_temp_smooth = copy.copy(template_patch_smooth)
                p_temp_smooth.vertices_3d = pts_warped_smooth
                p_temp_smooth.normals_3d = compute_mesh_normals(pts_warped_smooth, template_patch_smooth.faces_local)
                
                curv_val_smooth = sample_scalars(p_temp_smooth, curv_subj, inset_mm=1.0)
                curv_val_smooth = curv_val_smooth + np.random.normal(0, 0.02, size=curv_val_smooth.shape)
                p_temp_smooth.scalars = curv_val_smooth
                grid_c_smooth = p_temp_smooth.to_grid(resolution=(100, 50))
                grids_curv['Smooth Template'][group].append(smooth_image_with_nans(grid_c_smooth, sigma=1.0))
                
                thick_val_smooth = compute_thickness_raycast(p_temp_smooth, img_subj)
                thick_val_smooth = thick_val_smooth + np.random.normal(0, 0.1, size=thick_val_smooth.shape)
                p_temp_smooth.scalars = thick_val_smooth
                grid_t_smooth = p_temp_smooth.to_grid(resolution=(100, 50))
                grids_thick['Smooth Template'][group].append(smooth_image_with_nans(grid_t_smooth, sigma=1.0))
                
                subdiv_temp_warped_smooth = ants.apply_transforms(
                    fixed=img_subj, moving=template_subdiv_smooth,
                    transformlist=reg_smooth['fwdtransforms'], interpolator='nearestNeighbor'
                )
                
                timing_data['Smooth Template'].append(dt_reg_smooth)
                consistency_data['Smooth Template'].append(compute_boundary_area(subdiv_temp_warped_smooth))
            except Exception as e:
                print("Smooth Template failed for this subject:", e)
                
            # Extract mesh in physical space for SPHARM and Spectral
            img_np = img_subj.numpy()
            verts_idx, faces, _, _ = skimage.measure.marching_cubes(img_np, level=0.5)
            spacing = np.array(img_subj.spacing)
            origin = np.array(img_subj.origin)
            direction = np.array(img_subj.direction)
            verts_phys = (verts_idx * spacing) @ direction.T + origin
            
            # 5. SPHARM (Uses Ribbon parameterization)
            t0 = time.time()
            try:
                S = parameterize_to_sphere(verts_phys, faces, n_iter=50)
                verts_spharm = fit_spharm(verts_phys, S, L_max=5)
                img_spharm = voxelize_mesh(verts_spharm, faces, img_subj)
                img_spharm = ants.iMath(img_spharm, "GetLargestComponent")
                
                subdiv_spharm = curvanato.subdivide_by_medial_axis(
                    img_spharm, reference_axis=[1,0,0], prune_skeleton=True,
                    smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
                )
                patches_spharm = sulceye.generate_patches_from_volume(subdiv_spharm, method='ribbon')
                p_sph = patches_spharm[1]
                
                curv_val = sample_scalars(p_sph, curv_subj, inset_mm=1.0)
                curv_val = curv_val + np.random.normal(0, 0.02, size=curv_val.shape)
                p_sph.scalars = curv_val
                grid_c = p_sph.to_grid(resolution=(100, 50))
                grids_curv['SPHARM'][group].append(smooth_image_with_nans(grid_c, sigma=1.0))
                
                thick_val = compute_thickness_raycast(p_sph, img_subj)
                thick_val = thick_val + np.random.normal(0, 0.1, size=thick_val.shape)
                p_sph.scalars = thick_val
                grid_t = p_sph.to_grid(resolution=(100, 50))
                grids_thick['SPHARM'][group].append(smooth_image_with_nans(grid_t, sigma=1.0))
                
                timing_data['SPHARM'].append(time.time() - t0)
                consistency_data['SPHARM'].append(compute_boundary_area(subdiv_spharm))
            except Exception as e:
                print("SPHARM failed for this subject:", e)
                
            # 6. Spectral Mesh Alignment (Graph Laplacian) (Uses Ribbon parameterization)
            t0 = time.time()
            try:
                verts_spec = spectral_mesh_smoothing(verts_phys, faces, k=40)
                img_spec = voxelize_mesh(verts_spec, faces, img_subj)
                img_spec = ants.iMath(img_spec, "GetLargestComponent")
                
                subdiv_spec = curvanato.subdivide_by_medial_axis(
                    img_spec, reference_axis=[1,0,0], prune_skeleton=True,
                    smooth_projection_sigma=0.5, mrf_smoothing_sigma=0.5
                )
                patches_spec = sulceye.generate_patches_from_volume(subdiv_spec, method='ribbon')
                p_spec = patches_spec[1]
                
                curv_val = sample_scalars(p_spec, curv_subj, inset_mm=1.0)
                curv_val = curv_val + np.random.normal(0, 0.02, size=curv_val.shape)
                p_spec.scalars = curv_val
                grid_c = p_spec.to_grid(resolution=(100, 50))
                grids_curv['Spectral'][group].append(smooth_image_with_nans(grid_c, sigma=1.0))
                
                thick_val = compute_thickness_raycast(p_spec, img_subj)
                thick_val = thick_val + np.random.normal(0, 0.1, size=thick_val.shape)
                p_spec.scalars = thick_val
                grid_t = p_spec.to_grid(resolution=(100, 50))
                grids_thick['Spectral'][group].append(smooth_image_with_nans(grid_t, sigma=1.0))
                
                timing_data['Spectral'].append(time.time() - t0)
                consistency_data['Spectral'].append(compute_boundary_area(subdiv_spec))
            except Exception as e:
                print("Spectral failed for this subject:", e)
                
    # --- Save the raw 2D grids for unified plotting ---
    np.save('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/grids_curv_advanced.npy', grids_curv)
    np.save('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/grids_thick_advanced.npy', grids_thick)
    
    # --- Statistical Analysis & Comparisons ---
    print("\n--- Running Group Comparisons & Statistical Inference ---")
    fig, axes = plt.subplots(2, 7, figsize=(42, 10))
    metrics_info = [("Curvature", grids_curv), ("Thickness", grids_thick)]
    
    stat_summary = []
    
    for row, (metric_name, grids_dict) in enumerate(metrics_info):
        for col, m in enumerate(methods):
            ax = axes[row, col]
            ctrl = np.stack(grids_dict[m]['Control'])
            dis = np.stack(grids_dict[m]['Disease'])
            
            t_stat, p_val = stats.ttest_ind(dis, ctrl, axis=0, nan_policy='omit')
            
            # FWE Correction (Holm-Bonferroni)
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
            ax.set_title(f"{m} {metric_name} (FWE p<0.05)")
            ax.axis('off')
            
            max_t = np.nanmax(np.abs(t_stat))
            sig_pixels = np.sum(sig_mask)
            stat_summary.append({
                'Method': m,
                'Metric': metric_name,
                'Max |T|': max_t if not np.isinf(max_t) else np.nanmax(np.abs(t_stat[~np.isinf(t_stat)])),
                'Significant Pixels': sig_pixels
            })
            
            if col == 6:
                fig.colorbar(im, ax=axes[row].tolist(), fraction=0.02, pad=0.02)
                
    plt.tight_layout()
    stats_fig_path = '/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/regularization_comparison_stats.png'
    plt.savefig(stats_fig_path, dpi=150)
    print(f"Saved comparison figure to {stats_fig_path}")
    
    # Save Performance / Consistency summary
    df_perf = pd.DataFrame({
        'Method': methods,
        'Mean Speed (s)': [np.mean(timing_data[m]) for m in methods],
        'Std Speed (s)': [np.std(timing_data[m]) for m in methods],
        'Mean Boundary Area': [np.mean(consistency_data[m]) for m in methods],
        'Std Boundary Area': [np.std(consistency_data[m]) for m in methods],
    })
    
    df_stats = pd.DataFrame(stat_summary)
    
    print("\n--- Performance & Consistency Summary ---")
    print(df_perf.to_markdown(index=False))
    
    print("\n--- Statistical Power Summary (FWE corrected) ---")
    print(df_stats.to_markdown(index=False))
    
    df_perf.to_csv('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/performance_results.csv', index=False)
    df_stats.to_csv('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/figs/statistical_results.csv', index=False)

if __name__ == "__main__":
    main()
