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
ants.image_write(imggk,'/tmp/tempk.nii.gz')
ants.image_write(imgb,'/tmp/tempi.nii.gz')


kk=2
seg1=ants.threshold_image( imggk, kk, kk )
seg1b=seg1.clone()
for x in range(4):
    seg1b = antspyt1w.subdivide_labels(seg1b)

# now remove the "spine" or the skeleton of the curvature image
imgbkappa = curvanato.compute_curvature(imgb, smoothing=1.2, distance_map = True )
imgbkappaseg = ants.threshold_image( imgbkappa  + imgb, 'Otsu', 2 )
seg1b[ imgbkappaseg == 2 ] = 0
ants.image_write( imgbkappa, '/tmp/temp.nii.gz' )
ants.image_write( seg1b, '/tmp/temp2.nii.gz' )
