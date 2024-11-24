import ants
import curvanato
img=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
img = ants.crop_image( img, ants.iMath(img,'MD',20))
img = curvanato.compute_distance_map( img )
imggk=curvanato.cluster_image_gradient( img, n_clusters=4)
# np.unique(imggk.numpy())
# ants.image_write(imggk,'/tmp/tempk.nii.gz')
# ants.image_write(img,'/tmp/tempi.nii.gz')

print( curvanato.flatness( img  ))
mydt=curvanato.compute_distance_map( img )
skelt=curvanato.skeletonize_topo(img)
print( curvanato.flatness( skelt ) )
skel=curvanato.skeletonize(img)
ants.plot( img, skelt, crop=True )
