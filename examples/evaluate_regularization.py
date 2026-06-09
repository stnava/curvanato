import ants
import numpy as np
import time
import pandas as pd
import curvanato
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_boundary_area(subdiv):
    # The boundary is where Region 1 is adjacent to Region 2
    # Dilate region 1 by 1 voxel, and intersect with region 2
    r1 = ants.threshold_image(subdiv, 1, 1)
    r2 = ants.threshold_image(subdiv, 2, 2)
    
    r1_dilated = ants.iMath(r1, "MD", 1)
    boundary = r1_dilated * r2
    return boundary.sum()

def main():
    print("Loading base caudate data...")
    caudateAndVentricles = ants.image_read("curvanato/data/caud_vent.nii.gz")
    caudate = ants.threshold_image(caudateAndVentricles, 50, 50)
    caudate_crop = ants.crop_image(caudate, ants.iMath(caudate, "MD", 10))
    
    import siq
    print("Upsampling using siq.auto()...")
    base_caud_hr = siq.auto(caudate_crop)
    base_caud_hr = ants.threshold_image(base_caud_hr, 0.5, 1.5).threshold_image(1, 1)
    base_caud_hr = ants.iMath(base_caud_hr, "GetLargestComponent")

    methods = [
        {"name": "Baseline (None)", "kwargs": {"prune_skeleton": False, "smooth_projection_sigma": 0.0, "mrf_smoothing_sigma": 0.0}},
        {"name": "Prune Only", "kwargs": {"prune_skeleton": True, "smooth_projection_sigma": 0.0, "mrf_smoothing_sigma": 0.0}},
        {"name": "SP Only", "kwargs": {"prune_skeleton": False, "smooth_projection_sigma": 1.0, "mrf_smoothing_sigma": 0.0}},
        {"name": "MRF Only", "kwargs": {"prune_skeleton": False, "smooth_projection_sigma": 0.0, "mrf_smoothing_sigma": 1.0}},
        {"name": "Prune + SP", "kwargs": {"prune_skeleton": True, "smooth_projection_sigma": 1.0, "mrf_smoothing_sigma": 0.0}},
        {"name": "Prune + MRF", "kwargs": {"prune_skeleton": True, "smooth_projection_sigma": 0.0, "mrf_smoothing_sigma": 1.0}},
        {"name": "SP + MRF", "kwargs": {"prune_skeleton": False, "smooth_projection_sigma": 1.0, "mrf_smoothing_sigma": 1.0}},
        {"name": "Prune + SP + MRF", "kwargs": {"prune_skeleton": True, "smooth_projection_sigma": 1.0, "mrf_smoothing_sigma": 1.0}},
    ]

    results = []
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        print(f"Testing {method['name']}...")
        start_time = time.time()
        subdiv = curvanato.subdivide_by_medial_axis(base_caud_hr, reference_axis=[1,0,0], **method["kwargs"])
        elapsed = time.time() - start_time
        
        boundary_area = compute_boundary_area(subdiv)
        v1 = (subdiv.numpy() == 1).sum()
        v2 = (subdiv.numpy() == 2).sum()
        vol_ratio = v1 / (v2 + 1e-5)
        
        results.append({
            "Method": method["name"],
            "Boundary Area (Voxels)": boundary_area,
            "Vol Ratio (L/R)": vol_ratio,
            "Time (s)": elapsed
        })
        
        # Plot central slice for visual assessment
        r1 = subdiv.numpy() == 1
        r2 = subdiv.numpy() == 2
        # approximate boundary in Z slice
        boundary_np = np.logical_and(r1[1:] , r2[:-1]) 
        z_slice = np.argmax(boundary_np.sum(axis=(1,2)))
        if z_slice == 0: z_slice = subdiv.shape[0] // 2
        
        ax = axes[i]
        ax.imshow(subdiv.numpy()[z_slice, :, :], cmap='Set1', interpolation='nearest')
        ax.set_title(method["name"])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/regularization_eval.png', dpi=150)
    
    df = pd.DataFrame(results)
    print("\n--- Evaluation Results ---")
    print(df.to_markdown(index=False))
    
    with open("/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/eval_results.md", "w") as f:
        f.write("# Regularization Evaluation\n\n")
        f.write("We evaluated 8 combinations of regularization techniques to determine the optimal approach. A lower **Boundary Area** indicates a smoother cut. The **Vol Ratio (L/R)** shows how the method impacts the macroscopic volume balance.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n![Visual Comparison](/Users/stnava/.gemini/antigravity-cli/brain/48e3cac2-5675-4b9b-a9ed-5d1015a88987/regularization_eval.png)")

if __name__ == "__main__":
    main()
