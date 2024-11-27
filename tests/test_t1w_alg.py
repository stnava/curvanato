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
    citfn = re.sub( ".nii.gz", "_cit168.nii.gz" , fn )
    if not os.path.exists(citfn):
        t1b = antspynet.brain_extraction( t1, modality="t1threetissue" )['segmentation_image'].threshold_image(1,1)
        t1r = ants.rank_intensity( t1 * t1b )
#        antskm = ants.kmeans_segmentation(t1r, 2, kmask=t1b, mrf=0.1)['']
        cit = antspyt1w.deep_cit168( t1r, verbose=True)['segmentation']
        ants.image_write( cit, citfn)
    cit = ants.image_read( citfn )

##########################
segmentation = cit.clone()
prior_labels=[1,2]
prior_target_label=[2]
target_label=2
prior_labels=[3,4]
prior_target_label=[4]
target_label=18
labeled = segmentation * 0.0
curved = segmentation * 0.0
binaryimage = ants.threshold_image(segmentation, target_label, target_label).iMath("FillHoles").iMath("GetLargestComponent")
caud0 = curvanato.load_labeled_caudate(label=prior_labels, subdivide=0, option='laterality')
caudsd = curvanato.load_labeled_caudate(label=prior_target_label, subdivide=0, grid=0, option='laterality', binarize=True )
caudlat = curvanato.load_labeled_caudate(label=prior_labels, subdivide=0, grid=0, option='laterality', binarize=False )
prior_binary = caud0.clone() # ants.mask_image(caud0, caud0, prior_labels, binarize=True)
propagate=True
labeled, reg = curvanato.label_transfer( binaryimage, prior_binary, caudsd, propagate=propagate, regtx='SyN' )
labeledlat, reg = curvanato.label_transfer( binaryimage, prior_binary, caudlat, propagate=propagate, regtx='SyN', reg=reg )
smoothing=curvanato.compute_smoothing_spacing( binaryimage )
curvit = curvanato.compute_curvature( binaryimage, smoothing=smoothing, distance_map = True )
curvitr = ants.resample_image_to_target( curvit, labeled, interp_type='linear' )
binaryimager = ants.resample_image_to_target( binaryimage, labeled, interp_type='nearestNeighbor' )
imgd = curvanato.compute_distance_map( binaryimager )
imgg = curvanato.image_gradient( binaryimager )
ants.plot( binaryimager, labeledlat, axis=2, crop=True )
imggk=curvanato.cluster_image_gradient( binaryimager, binaryimager, 
    n_clusters=2, sigma=0.0, random_state=0 ) * binaryimager 
imggk = ants.iMath( binaryimager, "PropagateLabelsThroughMask", imggk, 200000, 0 )
ants.plot( binaryimager, labeled, crop=True, axis=2 )
sum2 = (ants.threshold_image(imggk,2,2) * labeled ).sum()
sum1 = (ants.threshold_image(imggk,1,1) * labeled ).sum()
if ( sum1 > sum2 ) :
    kmeansLabel=1 
else: 
    kmeansLabel=2

sidelabelRm2 = curvanato.remove_curvature_spine( curvitr, labeled )
sidelabelRm = curvanato.remove_curvature_spine( curvitr, 
    ants.threshold_image(imggk,kmeansLabel,kmeansLabel) )
labeled = ants.iMath( sidelabelRm * ants.threshold_image(imggk,kmeansLabel,kmeansLabel), 
        "PropagateLabelsThroughMask", 
        labeled * ants.threshold_image(imggk,kmeansLabel,kmeansLabel), 200000, 0 )
ants.plot( binaryimager, labeled, crop=True, axis=2 )
mydf = curvanato.make_label_dataframe( labeled )
labeled[ curvitr == 0 ] = 0.0
ants.image_write( curvitr, '/tmp/curvitr.nii.gz' )
ants.image_write( labeled, '/tmp/labeled.nii.gz' )
