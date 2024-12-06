# curvanato

local curvature quantification for anatomical data


### **Installation & Testing**

To install the package locally, navigate to the package root and run:

```bash
python3 -m pip install curvanato 
```

if you want to see some examples of different things the package can do, clone the source package and investigate the contents of the `tests` directory.

## Usage

```python
import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
import pandas as pd
import numpy as np
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "32"
ctype='fs'
tcaudR=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[2,4,6] )
caudseg=curvanato.load_labeled_caudate( option=ctype )
# this step is important - fs caudate segmentations look pretty terrible in comparison to what i am used to ....
caudseg=ants.threshold_image( caudseg, 50, 50 ).resample_image( [0.5,0.5,0.5], interp_type=0 ).threshold_image(0.5,1)
fn='/tmp/cc_example.nii.gz'
vlab=None
gr=0
subd=0
ccfn = [
        re.sub( ".nii.gz", "_"+ctype+"Rkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_"+ctype+"R.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rthk.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rkappa.csv" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rkappa.png" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rthk.png" , fn ) ]
pcaud=[3,4]
plabs=[4]
mytl=1
xx = curvanato.t1w_caudcurv(  caudseg, target_label=mytl, ventricle_label=vlab, 
        prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
        priorparcellation=tcaudR,  plot=True, smoothing=0.5,
        verbose=True )
ants.plot( xx[0], xx[1], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[4] )
ants.plot( xx[0], xx[2], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[5] )
for j in range(3):
    ants.image_write( xx[j], ccfn[j] )
xx[3].to_csv( ccfn[3] )
```


## Example data

this package has been tested on [ANTPD data from openneuro](https://openneuro.org/datasets/ds001907/versions/2.0.3).

could also try data [here](https://openneuro.org/datasets/ds004560/versions/1.0.1) which included repeated T1w acquisitions on same subjects but with different parameters.    however, last time i tried this, the link was not working.


```
rm -r -f build/ curvanato.egg-info/ dist/
python3 -m  build .
python3 -m pip install --upgrade twine
python3 -m twine upload --repository curvanato dist/*
```

