import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
import pandas as pd
import numpy as np
fn='.//bids/sub-RC4103/ses-1/anat/sub-RC4103_ses-1_T1w.nii.gz'
fn='.//bids/sub-RC4111/ses-1/anat/sub-RC4111_ses-1_T1w.nii.gz' # easy
fn='./bids//sub-RC4110/ses-2/anat/sub-RC4110_ses-2_T1w.nii.gz'
if os.path.exists(fn):
    t1=ants.image_read( fn )
    t1=ants.resample_image( t1, [0.5, 0.5, 0.5], use_voxels=False, interp_type=0 )
    hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn )
    if not os.path.exists(hoafn):
        hoa = antspynet.harvard_oxford_atlas_labeling(t1, verbose=True)['segmentation_image']
        ants.image_write( hoa, hoafn)
    hoa = ants.image_read( hoafn )


tcaud=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[1,3,5] )
vlab=None
leftside=True
gr=0
subd=0
if leftside:
    ccfn = [
        re.sub( ".nii.gz", "_caudLkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_caudL.nii.gz" , fn ),
        re.sub( ".nii.gz", "_caudLkappa.csv" , fn ) ]
    print("Begin " + fn + " caud kap")
    pcaud=[1,2]
    plabs=[2]
    xx = curvanato.t1w_caudcurv( t1, hoa, target_label=9, ventricle_label=vlab, 
        prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
        priorparcellation=tcaud, 
        verbose=True )
    for j in range(2):
        ants.image_write( xx[j], ccfn[j] )
    xx[2].to_csv( ccfn[2] )

labeled=xx[1]
curvitr=xx[0]
mydf = curvanato.make_label_dataframe( labeled )
descriptor = antspyt1w.map_intensity_to_dataframe( mydf, curvitr, labeled )
descriptor = curvanato.compute_geom_per_label( labeled, descriptor, curvanato.flatness, 'Flatness')
gdesc = ants.label_geometry_measures( xx[1] )
geos = ['Mean', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared', 'Eccentricity', 'Elongation','Flatness']
combined = pd.concat([descriptor, gdesc], ignore_index=True)
qq = curvanato.pd_to_wide( combined, column_values=geos)

otherside=True
if otherside:   
    tcaud=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[2,4,6] )
    pcaud=[3,4]
    plabs=[4]
    xx = curvanato.t1w_caudcurv( t1, hoa, target_label=10, ventricle_label=vlab, 
        prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr, 
        priorparcellation=tcaud, 
        verbose=True )
    ccfn = [
        re.sub( ".nii.gz", "_caudRkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_caudR.nii.gz" , fn ),
        re.sub( ".nii.gz", "_caudRkappa.csv" , fn ) ]
    ants.image_write( xx[0], ccfn[0] )
    ants.image_write( xx[1], ccfn[1] )
    xx[2].to_csv( ccfn[2] )

#
# test the jacobian    
# caud=ants.threshold_image( hoa, 9,9 )
# tcaud=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
# myj=curvanato.label_transfer( caud, tcaud, tcaud, jacobian=True )
#
