import ants
import antspynet
import antspyt1w
import curvanato
import re
import os  # For checking file existence
import pandas as pd
import numpy as np


def construct_path(filename, root_dir='./processedCSVSRFIRST/'):
    """
    Construct the path to each file of type "*deep_cit168lab.nii.gz"
    
    Parameters:
    filename (str): filename from the dataframe
    root_dir (str): root directory of the file system (default: './processedCSVSRFIRST/')
    
    Returns:
    str: constructed path to the file
    """
    # Split the filename into its components
    components = filename.split('-')
    # Extract the necessary components
    study_id, subject_id, date, scan_type, scan_id = components
    # Construct the path
    path = os.path.join(root_dir, 'PPMI', subject_id, date, f'{scan_type}Hierarchical', scan_id, f'{filename}-deep_cit168lab.nii.gz')
    path = re.sub("-T1w", "-T1wHierarchical",path)
    if os.path.isfile(path):
        return path
    else:
        return None



# Example usage:
filename = "PPMI-100018-20210202-T1w-1497578"
print(path)

ctype='cit'
tcaudL=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[1,3,5] )
tcaudR=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[2,4,6] )
vlab=None
leftside=True
gr=0
subd=0
otherside=True
mydf = pd.read_csv( "CNAsynNeg_and_PDSporadicAsynNeg_ids_PPMI_only.csv" )
for fn in list(mydf['filename']):
    path = construct_path(fn)
    if path is not None:
        print(  path )
        cit = ants.image_read( path )
        ccfn = [
            re.sub( ".nii.gz", "_"+ctype+"Rkappa.nii.gz" , path ), 
            re.sub( ".nii.gz", "_"+ctype+"R.nii.gz" , path ),
            re.sub( ".nii.gz", "_"+ctype+"Rkappa.csv" , path ) ]
        pcaud=[3,4]
        plabs=[4]
        if ctype == 'cit':
            mytl=18
        xx = curvanato.t1w_caudcurv( cit, target_label=mytl, ventricle_label=vlab, 
            prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
            priorparcellation=tcaudR,  plot=False,
            verbose=True )
        for j in range(2):
            ants.image_write( xx[j], ccfn[j] )
        xx[2].to_csv( ccfn[2] )
        ######
        mytl=2
        ccfn = [
            re.sub( ".nii.gz", "_"+ctype+"Lkappa.nii.gz" , path ), 
            re.sub( ".nii.gz", "_"+ctype+"L.nii.gz" , path ),
            re.sub( ".nii.gz", "_"+ctype+"Lkappa.csv" , path ) ]
        print("Begin " + fn + " caud kap")
        pcaud=[1,2]
        plabs=[2]
        xx = curvanato.t1w_caudcurv( cit, target_label=2, ventricle_label=vlab, 
            prior_labels=pcaud, prior_target_label=plabs, subdivide=subd, grid=gr,
            priorparcellation=tcaudL,  plot=True,
            verbose=True )
        for j in range(2):
            ants.image_write( xx[j], ccfn[j] )
        xx[2].to_csv( ccfn[2] )


