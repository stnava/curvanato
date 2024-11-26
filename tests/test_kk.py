import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
import pandas as pd
import numpy as np
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


caud=ants.threshold_image( hoa, 9, 9 )
segc = ants.crop_image( caud, ants.iMath(caud,'MD',5))
imgd = curvanato.compute_distance_map( segc )
imggk=curvanato.cluster_image_gradient( imgd, segc, n_clusters=2, sigma=0.25) * segc 
imggk = ants.iMath( segc, "PropagateLabelsThroughMask", imggk, 200000, 0 )
ants.plot( segc, imggk, axis=2 )
mykk = curvanato.shape_split_thickness( imggk, g=1, w=2, verbose=True )
ants.image_write( imggk, '/tmp/temp.nii.gz')
ants.image_write( mykk, '/tmp/tempk.nii.gz')
mykkbig=ants.decrop_image( mykk, t1*0 )
ants.image_write( mykkbig, '/tmp/tempkb.nii.gz')
mykkbig=ants.decrop_image( imggk, t1*0 )
ants.image_write( mykkbig, '/tmp/temparc.nii.gz')
#
# test the jacobian    
# caud=ants.threshold_image( hoa, 9,9 )
# tcaud=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
# myj=curvanato.label_transfer( caud, tcaud, tcaud, jacobian=True )
# ants.image_write( caud, '/tmp/individual.nii.gz' )
# ants.image_write( tcaud, '/tmp/template.nii.gz' )
# ants.image_write( myj, '/tmp/logj.nii.gz' )
