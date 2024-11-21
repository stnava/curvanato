import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
# ANTPD data
fn='./bids//sub-RC4110/ses-2/anat/sub-RC4110_ses-2_T1w.nii.gz'
fn='.//bids/sub-RC4111/ses-1/anat/sub-RC4111_ses-1_T1w.nii.gz' # easy
if os.path.exists(fn):
    t1=ants.image_read( fn )
    hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn )
    ccfn = [
        re.sub( ".nii.gz", "_caud.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_caudkappa.nii.gz" , fn ),
        re.sub( ".nii.gz", "_caudkappa.csv" , fn ) ]
    if not os.path.exists(hoafn):
        hoa = antspynet.harvard_oxford_atlas_labeling(t1, verbose=True)['segmentation_image']
        ants.image_write( hoa, hoafn)
    hoa = ants.image_read( hoafn )
    xx = curvanato.t1w_caudcurv( t1, hoa, target_label=9, prior_labels=[1, 2], 
        prior_target_label=[1,2], subdivide=0, grid=16, verbose=True )
    # xxx=ants.decrop_image( xx[1], t1 * 0.0 )
    ants.image_write( xx[0], ccfn[0] )
    ants.image_write( xx[1], ccfn[1] )
    xx[2].to_csv( ccfn[2] )
