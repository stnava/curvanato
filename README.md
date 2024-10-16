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


