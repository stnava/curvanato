import ants
import curvanato
img=curvanato.load_labeled_caudate( option='hmt', binarize=True, label=[1,3,5] )
print( curvanato.flatness( img  ))
mydt=curvanato.compute_distance_map( img )
skelt=curvanato.skeletonize_topo(img)
print( curvanato.flatness( skelt ) )
skel=curvanato.skeletonize(img)
ants.plot( img, skelt, crop=True )
