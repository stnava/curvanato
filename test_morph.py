import numpy as np
import ants
import curvanato
import sulceye
import plotly.graph_objects as go

base_caud = curvanato.load_labeled_caudate(option='laterality', binarize=True, label=[1,2])
dist = curvanato.compute_distance_map(base_caud)
coords = np.indices(base_caud.shape)
x_coords = coords[0]
bias = (x_coords - x_coords.mean()) * 0.05
bias_img = base_caud.new_image_like(bias)

noise = ants.make_image(base_caud.shape, np.random.normal(0, 1.0, base_caud.shape))
noise = ants.copy_image_info(base_caud, noise)
noise = ants.smooth_image(noise, 1.0)

dist_mod = dist + bias_img + noise
new_mask = ants.threshold_image(dist_mod, -100, 0)
print(base_caud.sum(), new_mask.sum())

