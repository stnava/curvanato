import ants
import curvanato
import numpy as np
img=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
imgb = ants.crop_image( img, ants.iMath(img,'MD',20))
imgd = curvanato.compute_distance_map( imgb )
imggk=curvanato.cluster_image_gradient( imgd, imgb, n_clusters=2, sigma=0.25) * imgb 
imggk = ants.iMath( imgb, "PropagateLabelsThroughMask", imggk, 200000, 0 )
print( np.unique(  imggk.numpy() ) )
ants.image_write(imggk,'/tmp/tempk.nii.gz')
ants.image_write(imgb,'/tmp/tempi.nii.gz')


imggk=curvanato.cluster_image_gradient( ants.iMath(imgb,'MaurerDistance'), imgb, n_clusters=6, sigma=1.5) * imgb 
imggk = ants.iMath( imgb, "PropagateLabelsThroughMask", imggk, 200000, 0 )
print( np.unique(  imggk.numpy() ) )
ants.image_write(imggk,'/tmp/tempk.nii.gz')
ants.image_write(imgb,'/tmp/tempi.nii.gz')
