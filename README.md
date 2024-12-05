# curvanato

local curvature quantification for anatomical data


### **Installation & Testing**

To install the package locally, navigate to the package root and run:

```bash
pip install .
```


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
# ANTPD data from open neuro
fn='.//bids/sub-RC4111/ses-1/anat/sub-RC4111_ses-1_T1w.nii.gz' # easy
fn='.//bids/sub-RC4103/ses-1/anat/sub-RC4103_ses-1_T1w.nii.gz'
fn='./bids//sub-RC4110/ses-2/anat/sub-RC4110_ses-2_T1w.nii.gz'
if os.path.exists(fn):
    t1=ants.image_read( fn )
    t1=ants.resample_image( t1, [0.5, 0.5, 0.5], use_voxels=False, interp_type=0 )
    hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn )
    if not os.path.exists(hoafn):
        hoa = antspynet.harvard_oxford_atlas_labeling(t1, verbose=True)['segmentation_image']
        ants.image_write( hoa, hoafn)
    hoa = ants.image_read( hoafn )
    citfn = re.sub( ".nii.gz", "_cit168.nii.gz" , fn )
    if not os.path.exists(citfn):
        t1b = antspynet.brain_extraction( t1, modality="t1threetissue" )['segmentation_image'].threshold_image(1,1)
        t1r = ants.rank_intensity( t1 * t1b )
        cit = antspyt1w.deep_cit168( t1r, verbose=True)['segmentation']
        ants.image_write( cit, citfn)
    cit = ants.image_read( citfn )

###################################################################################
###################################################################################
###################################################################################
###################################################################################
ctype='cit'
tcaudL=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[1,3,5] )
tcaudR=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[2,4,6] )
vlab=None
leftside=True
gr=0
subd=0
otherside=True
if otherside:
    ccfn = [
        re.sub( ".nii.gz", "_"+ctype+"Rkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_"+ctype+"R.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rthk.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rkappa.csv" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rkappa.png" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Rthk.png" , fn ) ]
    pcaud=[3,4]
    plabs=[4]
    if ctype == 'cit':
        mytl=18
    xx = curvanato.t1w_caudcurv(  cit, target_label=mytl, ventricle_label=vlab, 
        prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
        priorparcellation=tcaudR,  plot=True,
        verbose=True )
    ants.plot( xx[0], xx[1], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[4] )
    ants.plot( xx[0], xx[2], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[5] )
    for j in range(3):
        ants.image_write( xx[j], ccfn[j] )
    xx[3].to_csv( ccfn[3] )

if leftside:
    mytl=2
    ccfn = [
        re.sub( ".nii.gz", "_"+ctype+"Lkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_"+ctype+"L.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Lthk.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Lkappa.csv" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Lkappa.png" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"Lthk.png" , fn ) ]
    print("Begin " + fn + " caud kap")
    pcaud=[1,2]
    plabs=[2]
    xx = curvanato.t1w_caudcurv( cit, target_label=2, ventricle_label=vlab, 
        prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
        priorparcellation=tcaudL,  plot=True, searchrange=20,
        verbose=True )
    ants.plot( xx[0], xx[1], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[4] )
    ants.plot( xx[0], xx[1], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[5] )
    for j in range(3):
        ants.image_write( xx[j], ccfn[j] )
    xx[2].to_csv( ccfn[3] )



```


## Example data

this package has been tested on [ANTPD data from openneuro](https://openneuro.org/datasets/ds001907/versions/2.0.3).

could also try data [here](https://openneuro.org/datasets/ds004560/versions/1.0.1) which included repeated T1w acquisitions on same subjects but with different parameters.    however, last time i tried this, the link was not working.



