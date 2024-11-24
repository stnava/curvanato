import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
import pandas as pd
fn='./bids//sub-RC4110/ses-2/anat/sub-RC4110_ses-2_T1w.nii.gz'
fn='.//bids/sub-RC4103/ses-1/anat/sub-RC4103_ses-1_T1w.nii.gz'
fn='.//bids/sub-RC4111/ses-1/anat/sub-RC4111_ses-1_T1w.nii.gz' # easy
if os.path.exists(fn):
    t1=ants.image_read( fn )
    t1=ants.resample_image( t1, [0.5, 0.5, 0.5], use_voxels=False, interp_type=0 )
    hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn )
    if not os.path.exists(hoafn):
        hoa = antspynet.harvard_oxford_atlas_labeling(t1, verbose=True)['segmentation_image']
        ants.image_write( hoa, hoafn)
    segmentation = ants.image_read( hoafn )


prior_labels=[1,2]
prior_target_label=[2]
target_label=9
labeled = segmentation * 0.0
curved = segmentation * 0.0
binaryimage = ants.threshold_image(segmentation, target_label, target_label).iMath("FillHoles").iMath("GetLargestComponent")
caud0 = curvanato.load_labeled_caudate(label=prior_labels, subdivide=0, option='laterality')
caudsd = curvanato.load_labeled_caudate(label=prior_target_label, subdivide=3, grid=0 )
prior_binary = caud0.clone() # ants.mask_image(caud0, caud0, prior_labels, binarize=True)
propagate=True
labeled = curvanato.label_transfer( binaryimage, prior_binary, caudsd, propagate=propagate )
import numpy as np
spmag=0.0
spc=ants.get_spacing(segmentation)
for k in range(segmentation.dimension):
    spmag=spmag+spc[k]*spc[k]
####    
smoothing=np.sqrt( spmag )
curvit = curvanato.compute_curvature( binaryimage, smoothing=smoothing, distance_map = True )
curvitr = ants.resample_image_to_target( curvit, labeled, interp_type='linear' )
binaryimager = ants.resample_image_to_target( binaryimage, labeled, interp_type='nearestNeighbor' )
imgd = curvanato.compute_distance_map( binaryimager )
imggk=curvanato.cluster_image_gradient( binaryimager, binaryimager, n_clusters=2, sigma=0.25) * binaryimager 
imggk = ants.iMath( binaryimager, "PropagateLabelsThroughMask", imggk, 200000, 0 )
sum2 = (ants.threshold_image(imggk,2,2) * labeled ).sum()
sum1 = (ants.threshold_image(imggk,1,1) * labeled ).sum()
if ( sum1 > sum2 ) :
    kmeansLabel=1 
else: 
    kmeansLabel=2
sidelabelRm = curvanato.remove_curvature_spine( curvitr, 
    ants.threshold_image(imggk,kmeansLabel,kmeansLabel) )
labeled = ants.iMath( sidelabelRm * ants.threshold_image(imggk,kmeansLabel,kmeansLabel), 
        "PropagateLabelsThroughMask", 
        labeled * ants.threshold_image(imggk,kmeansLabel,kmeansLabel), 200000, 0 )
mydf = curvanato.make_label_dataframe( labeled )
labeled[ curvitr == 0 ] = 0.0
ants.image_write( curvitr, '/tmp/curvitr.nii.gz' )
ants.image_write( labeled, '/tmp/labeled.nii.gz' )
