import ants
import curvanato
import numpy as np
img=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
imgb = ants.crop_image( img, ants.iMath(img,'MD', 8))
kimage = curvanato.compute_curvature(imgb, smoothing=1.2, distance_map = True )
ants.image_write( kimage, '/tmp/temp.nii.gz' )
# imgd = ants.iMath(imgb,'MaurerDistance') * (-1.0 )
# kimage = ants.weingarten_image_curvature( imgd , 1.0)* imgb
