
# curvature.py
import numpy as np
import ants
import antspynet
import antspyt1w

def find_minimum(values):
    """
    Returns the minimum value from a list of numbers.
    
    Parameters:
    - values : list of float or int
        The list of numbers to find the minimum from.
        
    Returns:
    - float or int
        The minimum value in the list.
    """
    if not values:
        raise ValueError("The list is empty.")
        
    min_value = values[0]
    for val in values[1:]:
        if val < min_value:
            min_value = val
            
    return min_value

def create_spherical_volume(dim, radius, center):
    """
    Creates a 3D spherical segmentation within a volume.
    
    Parameters:
    - dim : tuple of int
        The dimensions of the 3D volume (x, y, z).
    - radius : float
        Radius of the sphere.
    - center : tuple of float
        Center of the sphere (x, y, z).
        
    Returns:
    - volume : ANTsImage
        A 3D binary volume with the sphere represented by 1s inside the sphere.
    """
    x, y, z = np.ogrid[:dim[0], :dim[1], :dim[2]]
    mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
    volume_data = np.zeros(dim)
    volume_data[mask] = 1  # Assign value of 1 to voxels inside the sphere
    
    # Convert the numpy array to an ANTsImage
    volume = ants.from_numpy(volume_data)
    return volume

import pkg_resources
import antspyt1w

def load_labeled_caudate(label=[1, 2], subdivide=0, grid=0, verbose=False):
    """
    Load a labeled NIfTI image of the caudate, mask it to specified labels, and optionally subdivide the labels.

    This function retrieves a pre-defined NIfTI file of the labeled caudate from the package data, 
    masks it to retain only the specified labels, and optionally subdivides these labels using the 
    `subdivide_labels` function from `antspyt1w`. The maximum number of labels is printed if `verbose` is set to `True`.

    Parameters:
    ----------
    label : list of int, optional
        A list of integer labels to retain in the mask. Default is [1, 2].
    subdivide : int, optional
        Number of times to subdivide the labels for finer segmentation. Default is 0.
    grid : int, optional
        Grid size with which to subdivide the labels for finer segmentation. Default is 0.
    verbose : bool, optional
        If `True`, prints the maximum label value after processing. Default is `False`.

    Returns:
    -------
    ants.ANTsImage
        A masked and optionally subdivided ANTsImage containing the labeled caudate.

    Examples:
    --------
    >>> from curvanato import load_labeled_caudate
    >>> seg_image = load_labeled_caudate(label=[1], subdivide=2, verbose=True)
    MaxDiv: 5
    >>> print(seg_image)
    ANTsImage (type: float)

    Notes:
    -----
    - The function assumes that the labeled caudate NIfTI file (`labeled_caudate.nii.gz`) 
      is present in the `data` directory of the `curvanato` package.
    - Requires the `pkg_resources`, `ants`, and `antspyt1w` packages.

    """
    # Get the path to the data file in the installed package
    nifti_path = pkg_resources.resource_filename(
        "curvanato", "data/labeled_caudate.nii.gz"
    )
    # Load the NIfTI file
    seg = ants.image_read(nifti_path)
    seg = ants.mask_image(seg, seg, label, binarize=True)
    if subdivide > 0:
        for x in range(subdivide):
            seg = antspyt1w.subdivide_labels(seg)
    elif grid > 0:
        gridder = tuple( [True] * seg.dimension )
        gg=ants.create_warped_grid( seg*0, grid_step=grid, grid_width=1, grid_directions=gridder )
        gg = gg * seg
        gg = ants.label_clusters( gg, 1 )
        if verbose:
            print("begin prop " + str( gg.max() )) 
        seg = ants.iMath(seg, 'PropagateLabelsThroughMask', gg, 1, 0)
        if verbose:
            print("end prop ") 
    if verbose:
        print("seg MaxDiv: " + str(seg.max()))
    return seg

def create_sine_wave_volume(dim, amplitude, frequency):
    """
    Creates a 3D volume with a sine wave pattern along one axis.
    
    Parameters:
    - dim : tuple of int
        The dimensions of the 3D volume (x, y, z).
    - amplitude : float
        Amplitude of the sine wave.
    - frequency : float
        Frequency of the sine wave.
        
    Returns:
    - volume : ANTsImage
        A 3D volume with a sine wave pattern.
    """
    x = np.linspace(0, dim[0], dim[0])
    y = np.linspace(0, dim[1], dim[1])
    z = np.linspace(0, dim[2], dim[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # Generate a sine wave along the x-axis
    sine_wave = amplitude * np.sin(frequency * X)
    # Create a binary volume by thresholding the sine wave pattern
    volume_data = (sine_wave > amplitude / 2).astype(float)
    # Convert the numpy array to an ANTsImage
    volume = ants.from_numpy(volume_data)
    return volume


def create_gaussian_bump_volume(dim, centers, sigma):
    """
    Creates a 3D volume with Gaussian-shaped bumps at specified centers.
    
    Parameters:
    - dim : tuple of int
        The dimensions of the 3D volume (x, y, z).
    - centers : list of tuple of float
        List of centers for each Gaussian bump.
    - sigma : float
        Standard deviation of the Gaussian (controls the spread of each bump).
        
    Returns:
    - volume : ANTsImage
        A 3D volume with Gaussian bumps.
    """
    x, y, z = np.ogrid[:dim[0], :dim[1], :dim[2]]
    volume_data = np.zeros(dim)
    for center in centers:
        cx, cy, cz = center
        gaussian_bump = np.exp(-((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / (2 * sigma**2))
        volume_data += gaussian_bump
    # Normalize to keep values between 0 and 1
    volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())
    volume = ants.from_numpy(volume_data)
    return volume


def compute_curvature(segmentation_image, smoothing=1.2, noise=[0, 0.01]):
    """
    Computes the curvature of a segmented anatomical structure.

    This function takes a binary 3D segmentation image and computes curvature values
    for each voxel in the segmented structure. The image can optionally be smoothed 
    and have noise added to mimic real-world imaging variability. 

    Parameters:
    - segmentation_image : numpy.ndarray
        A binary 3D segmentation image where the region of interest is represented by 
        foreground voxels.
    - smoothing : float, optional
        Smoothing factor applied to the curvature calculation, where higher values 
        increase smoothness (default is 1.2).
    - noise : list of [mean, std_dev], optional
        Parameters for additive Gaussian noise applied to the segmentation image, 
        specified as a list with mean and standard deviation (default is [0, 0.01]).

    Returns:
    - numpy.ndarray
        A 3D array with curvature values for each voxel in the segmented structure.
    
    Notes:
    - The `smoothing` parameter is used during the curvature calculation to control 
      the level of detail retained in the result.
    - The function relies on the ANTsPy library for image processing operations, 
      such as adding noise and smoothing.

    Example:
    ```python
    # Example usage
    segmentation_img = ants.image_read("segmentation.nii.gz")
    curvature_img = compute_curvature(segmentation_img)
    ants.plot(curvature_img)
    ```

    """
    # Add noise to the segmentation image to mimic variability
    segmentation_image_nz = ants.add_noise_to_image(segmentation_image, 'additivegaussian', noise)
    
    # Determine the minimum spacing for smoothing based on image resolution
    minspc = find_minimum(list(ants.get_spacing(segmentation_image)))
    
    # Smooth the image to prepare it for curvature computation
    spherical_volume_s = ants.smooth_image(segmentation_image_nz, minspc, sigma_in_physical_coordinates=True) 
    
    # Calculate the curvature image
    kimage = ants.weingarten_image_curvature(spherical_volume_s, smoothing)
    
    return kimage



def label_transfer(target_binary, prior_binary, prior_label, propagate=True ):
    """
    Perform label transfer from a prior image to a target image using deformable image registration.

    This function aligns a prior binary image to a target binary image using SyN registration (non-linear transformation) 
    and propagates the labels from the prior to the target. The transferred labels are post-processed to ensure 
    continuity using the 'PropagateLabelsThroughMask' operation from ANTs.

    Parameters:
    ----------
    target_binary : ants.ANTsImage
        A binary ANTsImage representing the target image.
    prior_binary : ants.ANTsImage
        A binary ANTsImage representing the prior image.
    prior_label : ants.ANTsImage
        A labeled ANTsImage representing the labels in the prior image.
    propagate: boolean

    Returns:
    -------
    ants.ANTsImage
        A labeled ANTsImage with the transferred labels mapped onto the target image.

    Examples:
    --------
    >>> import ants
    >>> from mypackage import label_transfer
    >>> target = ants.image_read("target_binary.nii.gz")
    >>> prior = ants.image_read("prior_binary.nii.gz")
    >>> prior_label = ants.image_read("prior_label.nii.gz")
    >>> result = label_transfer(target, prior, prior_label)
    >>> print(result)
    ANTsImage (type: integer)

    Notes:
    -----
    - The function uses SyN registration from ANTs to align images, which is computationally intensive.
    - Assumes that `target_binary` and `prior_binary` are binary masks, and `prior_label` is an integer-labeled image.
    - Requires the `ants` Python package for image processing.

    See Also:
    --------
    - `ants.registration`: For registration methods in ANTsPy.
    - `ants.apply_transforms`: To apply transformations to an image.
    - `ants.iMath`: For image-based mathematical operations.
    """
    target_binary_c = ants.crop_image( target_binary, ants.iMath( target_binary, "MD", 4 ) )
    reg = ants.registration(target_binary_c, prior_binary, 'SyNCC')
    labeled = ants.apply_transforms(target_binary_c, prior_label, reg['fwdtransforms'], 
        interpolator='nearestNeighbor' )
    if propagate:
        labeled = ants.iMath(target_binary_c, 'PropagateLabelsThroughMask', labeled, 1, 0)
    return labeled


def t1w_caudcurv(t1, segmentation, target_label=9, prior_labels=[1, 2], prior_target_label=2, subdivide=0, grid=0, propagate=True, verbose=False):
    """
    Perform caudate curvature mapping on a T1-weighted MRI image using prior labels for anatomical guidance.

    This function utilizes the Harvard-Oxford Atlas for initial labeling, processes specific target labels, 
    and transfers prior anatomical labels to compute curvature-related features of the caudate. The process 
    involves binary masking, curvature estimation, and label transfer using predefined or subdivided priors.

    Parameters:
    ----------
    t1 : ants.ANTsImage
        The T1-weighted MRI image to process.
    segmentation : ants.ANTsImage
        The segmentation associated with the T1
    target_label : int, optional
        The target label to isolate and process in the atlas segmentation. Default is 9.
    prior_labels : list of int, optional
        Labels from the prior segmentation that correspond to the caudate. Default is [1, 2].
    prior_target_label : int, optional
        The specific target label from the prior segmentation to transfer. Default is 2.
    subdivide : int, optional
        Number of subdivisions to apply to the prior target labels. Default is 0.
    grid : int, optional
        Number of grid divisions to apply to the prior target labels. Default is 0.
    propagate : boolean
    verbose : boolean

    Returns:
    -------
    ants.ANTsImage
        A labeled ANTsImage with curvature-adjusted labels mapped to the target.

    Examples:
    --------
    >>> import ants
    >>> from mypackage import t1w_caudcurv
    >>> t1_image = ants.image_read("subject_t1.nii.gz")
    >>> result = t1w_caudcurv(t1_image, target_label=9, prior_labels=[1, 2], subdivide=2)
    >>> print(result)
    ANTsImage (type: integer)

    Notes:
    -----
    - Requires `antspynet` for Harvard-Oxford atlas labeling and `ants` for image processing.
    - The function assumes the Harvard-Oxford atlas segmentation provides an appropriate 
      reference for caudate localization.
    - Prior caudate labels are loaded using the `load_labeled_caudate` function, and label 
      transfer is handled by the `label_transfer` function.

    See Also:
    --------
    - `antspynet.harvard_oxford_atlas_labeling`: Atlas-based segmentation function.
    - `load_labeled_caudate`: Loads labeled caudate images for prior segmentation.
    - `label_transfer`: Transfers labels between binary images.

    """
    # Target labels and prior labels correspond to left and right sides
    labeled = segmentation * 0.0
    curved = segmentation * 0.0
    binaryimage = ants.threshold_image(segmentation, target_label, target_label).iMath("FillHoles").iMath("GetLargestComponent")
    # FIXME compute curvature on the binary image
    caud0 = load_labeled_caudate(label=prior_labels, subdivide=0)
    if isinstance(prior_target_label, list):
        caudsd = load_labeled_caudate(label=prior_target_label, subdivide=subdivide, grid=grid )
    else:
        caudsd = load_labeled_caudate(label=[prior_target_label], subdivide=subdivide, grid=grid )
    if verbose:
        print('max caudsd '+ str( caudsd.max() ) )
    prior_binary = ants.mask_image(caud0, caud0, prior_labels, binarize=True)
    labeled = label_transfer( binaryimage, prior_binary, caudsd, propagate=propagate )
    return labeled

