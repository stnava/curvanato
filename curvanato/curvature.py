
# curvature.py
import numpy as np
import ants

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

def load_labeled_caudate(  label=[1,2], subdivide = 0, verbose=False ):
    # Get the path to the data file in the installed package
    nifti_path = pkg_resources.resource_filename(
        "curvanato", "data/labeled_caudate.nii.gz"
    )
    # Load the NIfTI file
    seg = ants.image_read(nifti_path)
    seg = ants.mask_image( seg, seg, label, binarize=True )
    for x in range( subdivide ):
        seg = antspyt1w.subdivide_labels( seg )
    if verbose:
        print("MaxDiv: " + str( seg.max() ) )
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

