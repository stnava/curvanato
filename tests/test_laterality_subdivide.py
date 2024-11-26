import ants
import curvanato
import numpy as np
seg=curvanato.load_labeled_caudate( option='hmt', binarize=False, label=[1,3,5] )
img=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
segc = ants.crop_image( seg, ants.iMath(img,'MD',20))
imgb = ants.crop_image( img, ants.iMath(img,'MD',20))
imgd = curvanato.compute_distance_map( imgb )
imggk=curvanato.cluster_image_gradient( imgd, imgb, n_clusters=2, sigma=0.25) * imgb 
imggk = ants.iMath( imgb, "PropagateLabelsThroughMask", imggk, 200000, 0 )
print( np.unique(  imggk.numpy() ) )
# mykk = curvanato.shape_split_thickness( imggk, g=2, w=1, verbose=True )
ants.image_write(imggk,'/tmp/tempk.nii.gz')
ants.image_write(imgb,'/tmp/tempi.nii.gz')
