
# test_curvature.py
import numpy as np
import ants
import antspynet
import curvanato

def test_compute_curvature(radius, smoo=1.0, distmap=False ):
    dim = (radius*2+5, radius*2+5, radius*2+5)
    center = (radius+2,radius+2,radius+2)
    spherical_volume = curvanato.create_spherical_volume(dim, radius, center)   
    if distmap:
        spherical_volume_s = curvanato.compute_distance_map( spherical_volume )# + spherical_volume
#        spherical_volume_s = ants.iMath( spherical_volume, 'MaurerDistance' )
    else:
        spherical_volume_s = ants.add_noise_to_image( spherical_volume, 'additivegaussian', [0,0.01] )
    spherical_volume_s = ants.smooth_image(spherical_volume_s, 1.0,sigma_in_physical_coordinates=True) 
    kvol = ants.weingarten_image_curvature( spherical_volume_s, smoo )
    # expected curvature is 1.0/r
    expected_k = 1.0 / float(radius)
    # evaluate the values at the surface
    dill=1
    spherical_volume_surf = spherical_volume - ants.iMath(spherical_volume,'ME',dill)
    computed_k = kvol[ spherical_volume_surf == 1 ].mean()
    spherical_volume_surf2 = ants.iMath(spherical_volume,'MD',1) - spherical_volume
    computed_k2 = kvol[ spherical_volume_surf2 == 1 ].mean()
    computed_k_mean = 0.5 * computed_k2 + 0.5 * computed_k
    print( "comp_1 : " + str(computed_k) + " comp_2 : " + str(computed_k2) + " comp_mu : " + str(computed_k_mean))
    print( "expected_k : " + str(expected_k))
    print( "comp_1 ratio : " + str(computed_k / expected_k ))
    print(  "comp_mu ratio : " + str(computed_k_mean / expected_k ) )
    return expected_k, computed_k_mean, computed_k, computed_k2
    # spherical_volume_s, spherical_volume_surf
    # assert result.shape == seg_image.shape
    # assert np.all(result == 0)


def test_gauss_bump_curvature(
    dim = (100, 100, 100),
    centers = [(50, 50, 50), (70, 70, 30), (30, 30, 70)],
    sigma = 10,
    smoo=1.0 ):
    spherical_volume = curvanato.create_gaussian_bump_volume(dim, centers, sigma)
    spherical_volume_s = ants.add_noise_to_image( spherical_volume, 'additivegaussian', [0,0.001] )
    spherical_volume_s = ants.smooth_image(spherical_volume_s, 1.0,sigma_in_physical_coordinates=True) 
    kvol = ants.weingarten_image_curvature( spherical_volume_s, smoo )
    return spherical_volume_s, kvol * ants.threshold_image( spherical_volume, 0.05, 1.e9) 


import pandas as pd

# Initialize an empty list to collect results
results = []

# List of radii
rlist = list( range(10,50,5) )
rlist.append( 60 )
rlist.append( 80 )

# Iterate over the radii
for r in rlist:
    print(f"Processing radius: {r}")
    truek, meank, k1, k2 = test_compute_curvature(r, distmap=False)
    truekd, meankd, k1d, k2d = test_compute_curvature(r, distmap=True)
    
    # Append the results to the list as a dictionary
    results.append({
        "Radius": r,
        "TrueK": truek,
        "MeanK": meank,
        "K1": k1,
        "K2": k2,
        "TrueKD": truekd,
        "MeanKD": meankd,
        "K1D": k1d,
        "K2D": k2d
    })

# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(results)

# Display the resulting DataFrame
print(results_df)

# Initialize a dictionary to store results
metrics_results = []

# Compute metrics for each "real" column vs other columns
real_columns = ["TrueK" ]
other_columns = ["MeanK", "K1", "K2", "MeanKD", "K1D", "K2D"]

for real_col in real_columns:
    for col in other_columns:
        # Compute correlation
        correlation = results_df[real_col].corr(results_df[col])
        
        # Compute Mean Absolute Error (MAE)
        mae = (results_df[real_col] - results_df[col]).abs().mean()
        
        # Compute Mean Absolute Percentage Error (MAPE)
        percentage_error = (
            (results_df[real_col] - results_df[col]).abs() / results_df[real_col].abs()
        ).mean() * 100
        
        # Append the metrics to the results list
        metrics_results.append({
            "Real_Column": real_col,
            "Compared_Column": col,
            "Correlation": correlation,
            "MAE": mae,
            "Percentage_Error (%)": percentage_error,
        })

# Convert the metrics results into a DataFrame
metrics_df = pd.DataFrame(metrics_results)

# Display the metrics DataFrame
print(metrics_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure a consistent theme
sns.set(style="whitegrid")

# Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    results_df["TrueK"], 
    results_df["K1D"], 
    c=results_df["Radius"], 
    cmap="viridis", 
    edgecolor="k", 
    s=100
)

# Add color bar to indicate radius
colorbar = plt.colorbar(scatter)
colorbar.set_label("Radius", fontsize=12)

# Set x and y limits to be the same
xy_min = min(results_df["TrueK"].min(), results_df["K1D"].min())
xy_max = max(results_df["TrueK"].max(), results_df["K1D"].max())
plt.xlim(xy_min, xy_max)
plt.ylim(xy_min, xy_max)

# Add a diagonal dotted line with slope 1
plt.plot([xy_min, xy_max], [xy_min, xy_max], color="black", linestyle="--", linewidth=1, label="y=x")

# Add labels, legend, and title
plt.xlabel("TrueK", fontsize=14)
plt.ylabel("MeanK", fontsize=14)
plt.title("Scatter Plot of TrueK vs K1D Colored by Radius", fontsize=16)
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
