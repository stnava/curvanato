import ants
import curvanato
img=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
imgb = ants.crop_image( img, ants.iMath(img,'MD',20))
imgd = curvanato.compute_distance_map( imgb )
imggk=curvanato.cluster_image_gradient( imgd, n_clusters=3, sigma=0.5) * imgb 
# np.unique(imggk.numpy())
# ants.image_write(imggk,'/tmp/tempk.nii.gz')
# ants.image_write(imgb,'/tmp/tempi.nii.gz')

print( curvanato.flatness( img  ))
mydt=curvanato.compute_distance_map( img )
skelt=curvanato.skeletonize_topo(img)
print( curvanato.flatness( skelt ) )
skel=curvanato.skeletonize(img)
ants.plot( img, skelt, crop=True )
