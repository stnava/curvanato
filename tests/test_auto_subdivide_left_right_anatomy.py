import ants
import curvanato
import numpy as np
#####
image = ants.image_read("~/.antspymm/PPMI_template0_deep_cit168lab.nii.gz")
#####
x1=6
xx=[x1,x1+16]
zz = ants.threshold_image( image, x1,x1)
rtx = curvanato.principal_axis_and_rotation( zz, [1,0,0])
zzz = ants.apply_ants_transform_to_image( rtx, zz, zz, interpolation='nearestNeighbor')
ants.plot( zzz, zzz, axis=2, crop=True )
# zzz=curvanato.align_to_y_axis( zz )
# ants.plot( zzz, crop=True, axis=2 )
output = curvanato.auto_subdivide_left_right_anatomy(
    image=image,
    label1=xx[0],
    label2=xx[1],
    symm_iterations=5,
    gradient_step=0.15,
    dilation_radius=16,
    partition_dilation=6,
    partition_axis=1,
    partition_k=3 )
#############################################
seg1b=ants.threshold_image(image,xx[0],xx[0])
seg2b=ants.threshold_image(image,xx[1],xx[1])
ants.plot(seg1b,output[0]*seg1b,axis=2,crop=True)
ants.plot(seg2b,output[1]*seg2b,axis=2,crop=True)
# ants.plot(output[2],axis=2,crop=True)
