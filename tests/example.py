import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence


import pandas as pd

def pd_to_wide(mydf, label_column='Description', column_values=None, prefix=""):
    """
    Transform a DataFrame to a wide format where the label_column values become column names 
    with a specified prefix and associated column_values.

    Parameters
    ----------
    mydf : pd.DataFrame
        Input DataFrame with labels and columns to pivot.
    
    label_column : str
        Column containing labels (e.g., 'Description') to use as new column names.
    
    column_values : list of str
        Column names to be included in the wide DataFrame.
    
    prefix : str
        Prefix to add before each label name in the new columns.

    Returns
    -------
    pd.DataFrame
        Wide format DataFrame.
    """
    if column_values is None:
        raise ValueError("Please specify column_values as a list of column names to include.")
    wide_df = pd.DataFrame()
    for col in column_values:
        temp_df = mydf[[label_column, col]].copy()
        temp_df.columns = [label_column, f"{prefix}{col}"]
        temp_df = temp_df.set_index(label_column).T
        wide_df = pd.concat([wide_df, temp_df], axis=1)
    wide_df.reset_index(drop=True, inplace=True)
    return wide_df


zeekee

# ANTPD data
fn='./bids//sub-RC4110/ses-2/anat/sub-RC4110_ses-2_T1w.nii.gz'
fn='.//bids/sub-RC4111/ses-1/anat/sub-RC4111_ses-1_T1w.nii.gz' # easy
if os.path.exists(fn):
    t1=ants.image_read( fn )
    t1=ants.resample_image( t1, [0.5, 0.5, 0.5], use_voxels=False, interp_type=0 )
    hoafn = re.sub( ".nii.gz", "_hoa.nii.gz" , fn )
    if not os.path.exists(hoafn):
        hoa = antspynet.harvard_oxford_atlas_labeling(t1, verbose=True)['segmentation_image']
        ants.image_write( hoa, hoafn)
    hoa = ants.image_read( hoafn )

leftside=True
if leftside:
    ccfn = [
        re.sub( ".nii.gz", "_caudLkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_caudL.nii.gz" , fn ),
        re.sub( ".nii.gz", "_caudLkappa.csv" , fn ) ]
    print("Begin " + fn + " caud kap")
    plabs=[1,2]
    xx = curvanato.t1w_caudcurv( t1, hoa, target_label=9, ventricle_label=1, 
        prior_labels=plabs, prior_target_label=plabs, subdivide=0, grid=16, verbose=True )
    ants.image_write( xx[0], ccfn[0] )
    ants.image_write( xx[1], ccfn[1] )
    xx[2].to_csv( ccfn[2] )


otherside=True
if otherside:   
    plabs=[3,4]
    xx = curvanato.t1w_caudcurv( t1, hoa, target_label=10, ventricle_label=2, 
        prior_labels=plabs, prior_target_label=plabs, subdivide=0, grid=16, verbose=True )
    ccfn = [
        re.sub( ".nii.gz", "_caudRkappa.nii.gz" , fn ), 
        re.sub( ".nii.gz", "_caudR.nii.gz" , fn ),
        re.sub( ".nii.gz", "_caudRkappa.csv" , fn ) ]
    ants.image_write( xx[0], ccfn[0] )
    ants.image_write( xx[1], ccfn[1] )
    xx[2].to_csv( ccfn[2] )
