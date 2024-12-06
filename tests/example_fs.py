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
caudseg=ants.threshold_image( caudseg, 50, 50 ).resample_image( [0.25,0.25,0.25], interp_type=0 ).threshold_image(0.5,1) 
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
        priorparcellation=tcaudR,  plot=True,
        verbose=True )
ants.plot( xx[0], xx[1], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[4] )
ants.plot( xx[0], xx[2], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[5] )
for j in range(3):
    ants.image_write( xx[j], ccfn[j] )
xx[3].to_csv( ccfn[3] )
##
##
##
ccfn = [
        re.sub( ".nii.gz", "_"+ctype+"RkappaSmoother.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_"+ctype+"RSmoother.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"RthkSmoother.nii.gz" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"RkappaSmoother.csv" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"RkappaSmoother.png" , fn ),
        re.sub( ".nii.gz", "_"+ctype+"RthkSmoother.png" , fn ) ]
xx = curvanato.t1w_caudcurv(  caudseg, target_label=mytl, ventricle_label=vlab, 
        prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
        priorparcellation=tcaudR,  plot=False, smoothing=2.4, # double
        verbose=True )
ants.plot( xx[0], xx[1], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[4] )
ants.plot( xx[0], xx[2], crop=True, axis=2, nslices=21, ncol=7, filename=ccfn[5] )
for j in range(3):
    ants.image_write( xx[j], ccfn[j] )
xx[3].to_csv( ccfn[3] )
