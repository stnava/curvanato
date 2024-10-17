# curvanato

local curvature quantification for anatomical data


### **Installation & Testing**

To install the package locally, navigate to the package root and run:

```bash
pip install .
```


## Usage

```python
import numpy as np
import curvananto

seg_image = np.zeros((10, 10, 10))  # Example segmentation
curvature = curvanato.compute_curvature(seg_image)
print(curvature)

```


## Example data

[here](https://openneuro.org/datasets/ds004560/versions/1.0.1) repeated T1w acquisitions on same subjects but with different parameters



