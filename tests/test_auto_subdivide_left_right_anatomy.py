############
import ants
import curvanato
import numpy as np
#####
image = ants.image_read("~/.antspymm/PPMI_template0_deep_cit168lab.nii.gz")
#####
# for x1 in list( range(1,16 ) ):
for x1 in list( range(6,7 ) ):
    xx=[x1,x1+16]
    #############################################
    seg1b=ants.threshold_image(image,xx[0],xx[0])
    seg2b=ants.threshold_image(image,xx[1],xx[1])
    if seg1b.max() > 0 and seg2b.max() > 0:
        print( " do " + str( x1) )
        output = curvanato.auto_subdivide_left_right_anatomy2(
            image=image,
            label1=xx[0],
            label2=xx[1],
            dilation_radius=12,
            partition_dilation=2,
            partition_axis=1,
            partition_k=5 )
        print( str(x1) + " auto-axis ")
        print( output[2] )
        ants.plot(seg1b,output[0],axis=2,crop=True)
        ants.plot(seg2b,output[1],axis=2,crop=True)

if False:
        output2 = curvanato.auto_subdivide_left_right_anatomy(
            image=image,
            label1=xx[0],
            label2=xx[1],
            dilation_radius=16,
            partition_dilation=1,
            partition_axis=1,
            partition_k=5, 
            reference_axis=[1,0,0] )
        ants.plot(seg1b,output2[0],axis=2,crop=True)
        ants.plot(seg2b,output2[1],axis=2,crop=True)
        print( str(x1) + " axis2 ")
