import ants
import curvanato
import numpy as np
seg=curvanato.load_labeled_caudate( option='laterality', binarize=False, label=[1,2] )
imgb=curvanato.load_labeled_caudate( option='laterality', binarize=True, label=[1,2] )
imgd = curvanato.compute_distance_map( imgb )
imggk1=curvanato.cluster_image_gradient( imgd, imgb, n_clusters=2, sigma=0.5) * imgb 
imggk1 = ants.iMath( imgb, "PropagateLabelsThroughMask", imggk1, 200000, 0 )
print( np.unique(  imggk1.numpy() ) )
ants.plot( imgb, imggk1, axis=2, crop=True )
ants.image_write(imggk1,'/tmp/tempk.nii.gz')
ants.image_write(imgb,'/tmp/tempi.nii.gz')

seg=curvanato.load_labeled_caudate( option='laterality', binarize=False, label=[3,4] )
imgb=curvanato.load_labeled_caudate( option='laterality', binarize=True, label=[3,4] )
imgd = curvanato.compute_distance_map( imgb )
imggk2=curvanato.cluster_image_gradient( imgd, imgb, n_clusters=2, sigma=0.5) * imgb 
imggk2 = ants.iMath( imgb, "PropagateLabelsThroughMask", imggk2, 200000, 0 )
ants.plot( imgb, imggk2, axis=2, crop=True )
# mykk = curvanato.shape_split_thickness( imggk, g=2, w=1, verbose=True )
ants.image_write(imggk2,'/tmp/tempk2.nii.gz')
ants.image_write(imgb,'/tmp/tempi2.nii.gz')

imggk=imggk2*0
imggk[ imggk1 == 1 ]=2
imggk[ imggk1 == 2 ]=1
imggk[ imggk2 == 1 ]=3
imggk[ imggk2 == 2 ]=4
ants.image_write( imggk, '/tmp/temp.nii.gz' )
