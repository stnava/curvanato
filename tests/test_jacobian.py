import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
import pandas as pd
import numpy as np
fn1='.//bids/sub-RC4111/ses-1/anat/sub-RC4111_ses-1_T1w.nii.gz' # easy
fn2='./bids//sub-RC4110/ses-2/anat/sub-RC4110_ses-2_T1w.nii.gz'
t1a=ants.image_read( fn1 )
hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn1 )
hoaa = ants.image_read( hoafn )
t1b=ants.image_read( fn2 )
hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn2 )
hoab = ants.image_read( hoafn )
cauda=ants.threshold_image( hoaa, 9, 9 )
caudb=ants.threshold_image( hoab, 9, 9 )
cauda = ants.crop_image( cauda, ants.iMath(cauda,'MD',20))
caudb = ants.crop_image( caudb, ants.iMath(caudb,'MD',20))
dcauda = curvanato.compute_distance_map( cauda )
dcaudb = curvanato.compute_distance_map( caudb )
reg=ants.registration( dcauda, dcaudb, 'SyN', syn_metric='CC', syn_sampling=2, verbose=1 )
jac=ants.create_jacobian_determinant_image( dcauda, reg['fwdtransforms'][0], 1 )
mygrid=ants.create_warped_grid( dcauda, grid_step=4, grid_width=2,
    grid_directions=(True,False,True), transform=reg['fwdtransforms'][0] )
ants.image_write( dcauda, '/tmp/dcauda.nii.gz' )
ants.image_write( dcaudb, '/tmp/dcaudb.nii.gz' )
ants.image_write( jac, '/tmp/jac.nii.gz' )
ants.image_write( mygrid, '/tmp/grid.nii.gz' )
ants.image_write( reg['warpedmovout'], '/tmp/dcaudbw.nii.gz' )
########################################################################
# myj=curvanato.label_transfer( caud, tcaud, tcaud, jacobian=True ) ####
# ants.image_write( caud, '/tmp/individual.nii.gz' ) 
# ants.image_write( tcaud, '/tmp/template.nii.gz' )
# ants.image_write( myj, '/tmp/logj.nii.gz' )
