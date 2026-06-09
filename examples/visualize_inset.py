import ants
import numpy as np
import curvanato
import sulceye
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import siq

def sample_curvature(p, curv, inset_mm=0.0):
    # Extract coordinates for mapping
    inv_dir = np.linalg.inv(curv.direction)
    origin = np.array(curv.origin)
    spacing = np.array(curv.spacing)
    
    # Inset the vertices
    inset_vertices = p.vertices_3d - (p.normals_3d * inset_mm)
    
    indices = ((inset_vertices - origin) @ inv_dir.T) / spacing
    indices = np.round(indices).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, curv.shape[0]-1)
    indices[:, 1] = np.clip(indices[:, 1], 0, curv.shape[1]-1)
    indices[:, 2] = np.clip(indices[:, 2], 0, curv.shape[2]-1)
    
    c_vals = curv.numpy()[indices[:, 0], indices[:, 1], indices[:, 2]]
    return c_vals

def main():
    print("Loading data...")
    caudateAndVentricles = ants.image_read("curvanato/data/caud_vent.nii.gz")
    base_caud = ants.threshold_image(caudateAndVentricles, 50, 50)
    base_caud = ants.crop_image(base_caud, ants.iMath(base_caud, "MD", 10))
    
    print("Upsampling...")
    base_caud_hr = siq.auto(base_caud)
    base_caud_hr = ants.threshold_image(base_caud_hr, 0.5, 1.5).threshold_image(1, 1)
    base_caud_hr = ants.iMath(base_caud_hr, "GetLargestComponent")
    
    print("Computing metrics...")
    dist = ants.iMath(base_caud_hr, "MaurerDistance")
    smooth_dist = ants.smooth_image(dist, 1.5)
    curv = ants.weingarten_image_curvature(smooth_dist, 1.5, 0)
    
    # Extract sulceye patch for the whole structure
    print("Extracting surface mesh...")
    patches = sulceye.generate_patches_from_volume(base_caud_hr)
    p = patches[1] # Whole structure
    
    # Sample at Surface (0mm)
    curv_surface = sample_curvature(p, curv, inset_mm=0.0)
    
    # Sample Inset (1mm)
    curv_inset = sample_curvature(p, curv, inset_mm=1.0)
    
    print("Generating Plotly comparison...")
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]],
        subplot_titles=("Surface Curvature (0mm Inset)", "Deep Curvature (1mm Inset)")
    )
    
    # Set shared coloraxis bounds for fair comparison
    vmin = min(np.percentile(curv_surface, 5), np.percentile(curv_inset, 5))
    vmax = max(np.percentile(curv_surface, 95), np.percentile(curv_inset, 95))
    
    # Surface Trace
    trace_surf = go.Mesh3d(
        x=p.vertices_3d[:, 0], y=p.vertices_3d[:, 1], z=p.vertices_3d[:, 2],
        i=p.faces_local[:, 0], j=p.faces_local[:, 1], k=p.faces_local[:, 2],
        intensity=curv_surface,
        colorscale='Viridis',
        cmin=vmin, cmax=vmax,
        showscale=False
    )
    fig.add_trace(trace_surf, row=1, col=1)
    
    # Inset Trace
    trace_inset = go.Mesh3d(
        x=p.vertices_3d[:, 0], y=p.vertices_3d[:, 1], z=p.vertices_3d[:, 2],
        i=p.faces_local[:, 0], j=p.faces_local[:, 1], k=p.faces_local[:, 2],
        intensity=curv_inset,
        colorscale='Viridis',
        cmin=vmin, cmax=vmax,
        colorbar=dict(title="Curvature"),
        showscale=True
    )
    fig.add_trace(trace_inset, row=1, col=2)
    
    fig.update_layout(title_text="Weingarten Curvature: Surface vs. Inset Normal Sampling")
    fig.write_html("/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/inset_comparison.html")
    print("Saved interactive comparison to inset_comparison.html")

if __name__ == "__main__":
    main()
