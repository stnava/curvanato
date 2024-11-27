
# curvature.py
import numpy as np
import ants
import antspynet
import antspyt1w
import pandas as pd
import pandas as pd
import ants


import numpy as np
import ants
from sklearn.decomposition import PCA

def flatness(binary_image):
    """
    Compute the flatness of binary image using PCA on voxel positions.

    Parameters
    ----------
    binary_image : ants.ANTsImage
        Input binary image.

    Returns
    -------
    flatness : float
        Ratio of the smallest to the largest PCA eigenvalue (flatness metric).
    """
    # Get voxel positions where intensity > threshold
    voxel_positions = np.argwhere(binary_image.numpy() > 0)
    
    # Perform PCA on voxel positions
    pca = PCA(n_components=3)
    pca.fit(voxel_positions)
    
    # Eigenvalues from PCA
    eigenvalues = pca.explained_variance_
    
    # Flatness ratio (smallest eigenvalue / largest eigenvalue)
    flatness = eigenvalues.min() / eigenvalues.max()
    
    return 1.0 - flatness

def skeletonize_topo(x, laplacian_threshold=0.30, propagation_option=2 ):
    """
    Skeletonize a binary segmentation with topology preserving methods

    This function uses topologically constrained label propagation to thin
    an existing segmentation into a skeleton. It works best when the input
    segmentation is well-composed.

    Parameters
    ----------
    x : ANTsImage
        Input binary image.

    laplacian_threshold : float, optional
        Threshold for the Laplacian speed image, between 0 and 1 (default is 0.25).

    propagation_option : int, optional
        Propagation constraint option:
        - 0: None
        - 1: Well-composed
        - 2: Topological constraint (default is 1).

    Returns
    -------
    ANTsImage
        Skeletonized binary image.
    """
    # Negate image to prepare for processing
    wm = x.clone()# ants.threshold_image(x, 0, 0)

    # Compute Laplacian of the binary image
    wmd = ants.iMath(x, "MaurerDistance") * x  # Distance transform
    wmd[ wm == 1] = wmd[ wm == 1] * ( -1.0 )
    wmdl = ants.iMath(wmd, "Laplacian", 1.0, 1)
#    ants.plot( wm, crop=True )
#    ants.plot( wmd, crop=True  )
#    ants.plot( wm, wmdl, crop=True  )
    # Threshold to create a speed image
    speed = ants.threshold_image(wmdl, 0.0, laplacian_threshold)
    speed = ants.threshold_image(speed, 0, 0)  # Negate the speed image
    # Extract the largest connected component
    wm = ants.iMath(wm, "GetLargestComponent")
    # Propagate labels through the speed image using topological constraints
    wmNeg = x * 0.0
    wmNeg[ x == 0 ]=1
    wmskel = ants.iMath(speed, "PropagateLabelsThroughMask", wmNeg, 200000, propagation_option)
    wmskel = ants.threshold_image(wmskel, 0, 0)  # Final negation
    return wmskel

def skeletonize( x ):
    import numpy as np
    from skimage.morphology import skeletonize_3d
    import nibabel as nib  # For loading and saving NIfTI images
    # Ensure the image is binary (0s and 1s)
    binary_image = (x.numpy() > 0).astype(np.uint8)
    # Apply 3D skeletonization
    skeleton = ants.from_numpy( skeletonize_3d(binary_image) )
    return ants.copy_image_info( x, skeleton )


def compute_distance_map(binary_image):
    """
    Compute the signed distance map of a binary image using the Maurer distance transform.
    
    The function calculates the distance of each pixel/voxel in the binary image to the nearest 
    boundary, with negative distances assigned inside the binary object.

    Parameters
    ----------
    binary_image : ants.ANTsImage
        A binary image where foreground pixels are 1 and background pixels are 0.

    Returns
    -------
    distance_map : ants.ANTsImage
        The computed signed distance map, where negative values represent distances 
        inside the binary object and positive values represent distances outside.
    
    Example
    -------
    >>> binary_image = ants.threshold_image(ants.image_read("example.nii.gz"), 0.5, 1.0)
    >>> distance_map = compute_distance_map(binary_image)
    >>> ants.image_write(distance_map, "distance_map.nii.gz")
    """
    # Clone the input image for manipulation
    binary_clone = binary_image.clone()

    # Compute the Maurer distance transform
    distance_transform = ants.iMath(binary_image, "MaurerDistance") * (-1.0 )
    distance_transform[binary_clone == 1] += 1.0

    # Assign negative distances inside the binary region
#    signed_distance = distance_transform * binary_image
#    signed_distance[binary_clone == 1] *= -1.0

    return distance_transform * binary_clone

def make_label_dataframe(label_image):
    """
    Generate a pandas DataFrame with unique label values (excluding 0) from a label image and their descriptions.

    Parameters:
    ----------
    label_image : ants.ANTsImage
        The input label image containing integer labels.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with two columns: 'Label' and 'Description'. Each row corresponds to a unique label, excluding 0.
    """
    # Extract unique label values from the image and convert to integers
    unique_labels = list(set(label_image.numpy().flatten()))
    unique_labels = [int(label) for label in unique_labels if label != 0]
    df = pd.DataFrame({
        'Label': unique_labels,
        'Description': ["Label" + str(label) for label in unique_labels]
    })
    return df


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

def load_labeled_caudate(label=[1, 2], subdivide=0, grid=0, option='laterality', binarize=True, verbose=False):
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
    option: string
        either laterality or hmt (head, midbody, tail)
    binarize : bool, optional
        If `True`, binarizes the image. Default is True.
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
    if option == 'laterality' :
        nifti_path = pkg_resources.resource_filename(
            "curvanato", "data/labeled_caudate_medial_vs_lateral.nii.gz"
        )
    else:
        nifti_path = pkg_resources.resource_filename(
            "curvanato", "data/labeled_caudate_head_mid_tail.nii.gz"
        )
    # Load the NIfTI file
    seg = ants.image_read(nifti_path)
    seg = ants.mask_image(seg, seg, label, binarize=binarize)
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
        seg = ants.iMath(seg, 'PropagateLabelsThroughMask', gg, 1000, 0)
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


def pd_to_wide(mydf, label_column='Description', column_values=None, prefix=""):
    """
    Transform a DataFrame to a wide format where each label in the label_column
    contributes to new column names, combined with the specified column_values.

    Parameters
    ----------
    mydf : pd.DataFrame
        Input DataFrame with labels and columns to pivot.
    
    label_column : str
        Column containing labels (e.g., 'Description') to use as new column names.
    
    column_values : list of str
        Column names to be included in the wide DataFrame.
    
    prefix : str
        Prefix to add before each label name in the new columns.

    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with one row and new column names.
    """
    if column_values is None:
        raise ValueError("Please specify column_values as a list of column names to include.")
    
    # Initialize an empty dictionary to hold the wide data
    wide_data = {}
    
    # Iterate over each column value and create new column names
    for col in column_values:
        for _, row in mydf.iterrows():
            new_col_name = f"{prefix}{row[label_column]}{col}"
            wide_data[new_col_name] = row[col]
    
    # Convert the dictionary to a one-row DataFrame
    wide_df = pd.DataFrame([wide_data])
    return wide_df

def compute_curvature(segmentation_image, smoothing=1.2, noise=[0, 0.01], distance_map = True ):
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
    - distance_map : boolean ( default  True)

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
    if distance_map:
        segmentation_image_nz = ants.iMath( segmentation_image, 'MaurerDistance' ) * (-1.0) # compute_distance_map( segmentation_image )
    
    # Determine the minimum spacing for smoothing based on image resolution
    minspc = find_minimum(list(ants.get_spacing(segmentation_image)))
    
    # Smooth the image to prepare it for curvature computation
    spherical_volume_s = ants.smooth_image(segmentation_image_nz, minspc, sigma_in_physical_coordinates=True) 
    
    # Calculate the curvature image
    kimage = ants.weingarten_image_curvature( spherical_volume_s , smoothing) * segmentation_image
    
    return kimage


def image_gradient(image, sigma = 0.25 ):
    """
    Computes the gradient of an ANTs image in all spatial dimensions.

    Parameters:
    - image: ANTsPy image. Input image for gradient computation.
    - sigma: physical coordinate sigma for smoothing the gradient

    Returns:
    - gradient_dict: Dictionary of ANTs images with keys as spatial dimensions ('x', 'y', 'z').
    """
    # Ensure image is valid
    if not isinstance(image, ants.ANTsImage):
        raise ValueError("Input must be an ANTsPy image.")

    # Get image dimensions
    dim = image.dimension
    spacing = image.spacing

    # Convert image to NumPy array for processing
    image_np = image.numpy()

    # Initialize dictionary to store gradient components
    gradient_dict = {}

    # Compute gradient along each dimension
    for axis in range(dim):
        # Compute finite differences along the current axis
        gradient_axis = np.gradient(image_np, spacing[axis], axis=axis)
        
        # Convert gradient back to ANTs image
        gradient_image = ants.from_numpy(gradient_axis, origin=image.origin, spacing=image.spacing)
        gradient_image = ants.smooth_image( gradient_image, sigma, sigma_in_physical_coordinates=True )
        
        # Add to dictionary with axis labels ('x', 'y', 'z')
        gradient_dict[chr(120 + axis)] = gradient_image  # chr(120) = 'x', chr(121) = 'y', etc.

    return gradient_dict

def cluster_image_gradient(image, binary_image, n_clusters=2, sigma=0.5, random_state=None):
    """
    Computes the gradient of an image and performs k-means clustering on the gradient.
    
    Parameters:
    - image: ANTsPy image. Input image to process.  could be a distance map.
    - binary_image: ANTsPy image. Input image to process. binary.
    - n_clusters: int. Number of clusters for k-means.
    - sigma: physical coordinate sigma for smoothing the gradient
    - random_state: int or None. Random state for reproducibility.
    
    Returns:
    - clustered_image: ANTsPy image with cluster labels.
    """
    import ants
    import numpy as np
    from sklearn.cluster import KMeans
    # Compute the gradient of the image
    gradient = image_gradient(image,sigma=sigma)
    
    # Extract gradient components as NumPy arrays
    gradient_x = gradient['x'].numpy()
    gradient_y = gradient['y'].numpy()
    if image.dimension == 3:
        gradient_z = gradient['z'].numpy()
    
    # Stack gradient components into a feature matrix
    if image.dimension == 2:
        feature_matrix = np.stack([gradient_x.ravel(), gradient_y.ravel()], axis=1)
    elif image.dimension == 3:
        feature_matrix = np.stack([gradient_x.ravel(), gradient_y.ravel(), gradient_z.ravel()], axis=1)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters+1, random_state=random_state,n_init='auto')
    labels = kmeans.fit_predict(feature_matrix)
    
    # Reshape labels to match the image shape
    label_image_np = labels.reshape(image.shape)
    
    # Convert the label array back to an ANTs image
    # clustered_image = ants.from_numpy(label_image_np.astype(np.float32) )
    clustered_image = ants.from_numpy( label_image_np )
    clustered_image = ants.copy_image_info( image, clustered_image )
    return clustered_image


def cluster_image_gradient_prior(image, binary_image, prior_label_image, n_clusters=2, sigma=0.5, random_state=None):
    """
    Computes the gradient of an image and performs k-means clustering on the gradient,
    initializing cluster centers based on a prior label image.

    Parameters:
    - image: ANTsPy image. Input image to process. Could be a distance map.
    - binary_image: ANTsPy image. Input binary mask image.
    - prior_label_image: ANTsPy image. Label image with unique labels greater than zero for initialization.
    - n_clusters: int. Number of clusters for k-means. Must match the number of unique labels in prior_label_image.
    - sigma: float. Physical coordinate sigma for smoothing the gradient.
    - random_state: int or None. Random state for reproducibility.

    Returns:
    - clustered_image: ANTsPy image with cluster labels.
    """
    import ants
    import numpy as np
    from sklearn.cluster import KMeans

    # Compute the gradient of the image
    gradient = image_gradient(image, sigma=sigma)
    
    # Extract gradient components as NumPy arrays
    gradient_x = gradient['x'].numpy()
    gradient_y = gradient['y'].numpy()
    if image.dimension == 3:
        gradient_z = gradient['z'].numpy()
    
    # Stack gradient components into a feature matrix
    if image.dimension == 2:
        feature_matrix = np.stack([gradient_x.ravel(), gradient_y.ravel()], axis=1)
    elif image.dimension == 3:
        feature_matrix = np.stack([gradient_x.ravel(), gradient_y.ravel(), gradient_z.ravel()], axis=1)

    # Mask the feature matrix using the binary image
    binary_mask = binary_image.numpy() > 0
    feature_matrix = feature_matrix[binary_mask.ravel()]
    
    # Compute cluster initialization centers from the prior label image
    prior_labels = prior_label_image.numpy()
    if prior_labels.shape != binary_mask.shape:
        raise ValueError("Shape of prior_label_image must match the binary_image and input image.")
    
    unique_labels = np.unique(prior_labels[prior_labels > 0])
    if len(unique_labels) != n_clusters:
        raise ValueError(f"Number of unique labels in prior_label_image ({len(unique_labels)}) "
                         f"does not match n_clusters ({n_clusters}).")
    
    centers = []
    binary_indices = np.where(binary_mask.ravel())[0]  # Get indices of the binary mask
    for label in unique_labels:
        # Get indices corresponding to the current label within the binary mask
        label_indices = binary_indices[prior_labels.ravel()[binary_indices] == label]
        if len(label_indices) > 0:
            label_coords = feature_matrix[np.isin(np.arange(len(feature_matrix)), label_indices)]
            # Compute the mean coordinate for this label
            centers.append(np.mean(label_coords, axis=0))
        else:
            raise ValueError(f"No valid data found in prior_label_image for label {label}.")
    centers = np.array(centers)

    # Perform k-means clustering with initialized centers
    kmeans = KMeans(n_clusters=n_clusters, init=centers, random_state=random_state, n_init=1)
    labels = kmeans.fit_predict(feature_matrix)
    
    # Create a full label image, filling in the background with zeros
    full_label_image_np = np.zeros_like(prior_labels, dtype=int)
    full_label_image_np[binary_mask] = labels + 1  # Add 1 to ensure labels are positive
    
    # Convert the label array back to an ANTs image
    clustered_image = ants.from_numpy(full_label_image_np, origin=image.origin, spacing=image.spacing)
    clustered_image = ants.copy_image_info(image, clustered_image)
    return clustered_image


def label_transfer(target_binary, prior_binary, prior_label, propagate=True, jacobian=False, regtx="antsRegistrationSyNQuickRepro[s]", reg=None ):
    """
    Perform label transfer from a prior image to a target image using deformable image registration.

    This function aligns a prior binary image to a target binary image using SyN registration (non-linear transformation) on the distance maps
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
    jacobian: boolean
        will return the jacobian of the registration to the prior space
    regtx : type of transform
    reg : optional existing registration result default None

    Returns:
    -------
    ants.ANTsImage
        A labeled ANTsImage with the transferred labels mapped onto the target image and a registration result

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
#    bindist = compute_distance_map( target_binary_c )
#    priork = compute_distance_map( prior_binary )
    smoothing=compute_smoothing_spacing( target_binary )
    bindist = compute_curvature( target_binary_c, smoothing=smoothing, distance_map = True  ) + target_binary_c
    priork = compute_curvature( prior_binary, smoothing=smoothing, distance_map = True  ) + prior_binary
    if jacobian:
        croppedTdog = ants.crop_image( priork, ants.iMath( prior_binary, "MD", 10 ) )
        reg = ants.registration( croppedTdog, bindist, regtx )
        return ants.create_jacobian_determinant_image( prior_binary, reg['fwdtransforms'][0],1 )
    if reg is None:
        reg = ants.registration(bindist, priork, regtx )
    labeled = ants.apply_transforms(target_binary_c, prior_label, reg['fwdtransforms'], 
        interpolator='nearestNeighbor' )
    if propagate:
        labeled = ants.iMath(target_binary_c, 'PropagateLabelsThroughMask', labeled, 1, 0)
    return labeled, reg


def remove_curvature_spine( curvature_image, segmentation_image, dilation=0 ):
    """
    Removes the spine region from an image based on curvature.

    Parameters:
    - curvature_image: ANTsPy image of curvature.
    - segmentation_image: ants image. from which to remove the spine.
    - dilation : int default zero

    Returns:
    - modified_segmentation_image: ANTsPy image with the spine region removed.
    """
    curvature_segmentation = ants.threshold_image(curvature_image + segmentation_image, "Otsu", 2 )
    curvature_segmentation = ants.threshold_image( curvature_segmentation, 2, 2 )
    if dilation > 0 :
        curvature_segmentation = ants.iMath( curvature_segmentation, "MD", dilation )
    modified_image = segmentation_image.clone()  # Clone the input image to avoid modifying it in place
    modified_image[ curvature_segmentation == 1 ] = 0
    return modified_image

def t1w_caudcurv( segmentation, target_label=9, ventricle_label=None, prior_labels=[1, 2], prior_target_label=2,  subdivide=0, grid=0, smoothing=None, propagate=True, priorparcellation=None, searchrange=25, plot=False, verbose=False):
    """
    Perform caudate curvature mapping on a caudate segmentation using prior labels for anatomical guidance.

    This function utilizes an initial labeling, processes specific target labels, 
    and transfers prior anatomical labels to compute curvature-related features of the caudate. The process 
    involves binary masking, curvature estimation, and label transfer using predefined or subdivided priors.

    Parameters:
    ----------
    segmentation : ants.ANTsImage
        The segmentation to process
    target_label : int, optional
        The target label to isolate and process in the atlas segmentation. Default is 9.
    ventricle_label : int or None, optional
        The target label that defines the ventricles. Default is 9.
    prior_labels : list of int, optional
        Labels from the prior segmentation that correspond to the caudate. Default is [1, 2].
    prior_target_label : int, optional
        The specific target label from the prior segmentation to transfer. Default is 2.
    subdivide : int, optional
        Number of subdivisions to apply to the prior target labels. Default is 0.
    grid : int, optional
        Number of grid divisions to apply to the prior target labels. Default is 0.
    priorparcellation : ants.image in the same space as the priors
        prior labels to map to the final parcellation default None.
    smoothing : float, optional
        Smoothing factor applied to the curvature calculation, where higher values 
        increase smoothness (default is the magnitude of the resolution of the image).
    propagate : boolean
    searchrange : number of kmeans searches default 25; set to zero to just use image registration
    plot : boolean
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
    if verbose:
        print("Begin")
    labeled = segmentation * 0.0
    curved = segmentation * 0.0
    binaryimage = ants.threshold_image(segmentation, target_label, target_label).iMath("FillHoles").iMath("GetLargestComponent")
    caud0 = load_labeled_caudate(label=prior_labels, 
        subdivide=0, grid=0, option='laterality', binarize=True)
    caudsd = load_labeled_caudate(label=prior_target_label, 
        subdivide=0, grid=0, option='laterality', binarize=True )
    # caudlat = load_labeled_caudate(label=prior_target_label, 
    #    subdivide=0, grid=0, option='laterality', binarize=False )
    if verbose:
        print("loaded")
    prior_binary = caud0.clone() # ants.mask_image(caud0, caud0, prior_labels, binarize=True)
    propagate=True
    if verbose:
        print("reg")
    if plot:
        ants.plot( binaryimage, binaryimage, crop=True, axis=2 )
        ants.plot( prior_binary, prior_binary, crop=True, axis=2 )
    regtx="antsRegistrationSyNQuickRepro[s]"
    regtx='SyN'
    labeled, reg = label_transfer( binaryimage, prior_binary, caudsd, propagate=propagate, regtx=regtx )
    if verbose:
        print("reg fin ")
    if plot:
        ants.plot( binaryimage, labeled, axis=2, crop=True )
        # ants.plot( binaryimage, labeledlat, axis=2, crop=True )
    # labeledlat, reg = label_transfer( binaryimage, prior_binary, caudlat, propagate=propagate, regtx='SyN',reg=reg)
    if verbose:
        print(" ... now curv")
    if smoothing is None:
        smoothing=compute_smoothing_spacing( binaryimage )
    curvit = compute_curvature( binaryimage, smoothing=smoothing, distance_map = True )
    curvitr = ants.resample_image_to_target( curvit, labeled, interp_type='linear' )
    if verbose:
        print("curv fin")
    binaryimager = ants.resample_image_to_target( binaryimage, labeled, interp_type='nearestNeighbor' )
    if searchrange > 0 :
        bestsum=0
        bestvol=9.e14
        labeledTemp = remove_curvature_spine( curvitr, labeled )
        for myrandstate in list(range(searchrange)):
            isbest=False
            imggk=cluster_image_gradient( binaryimager, binaryimager, n_clusters=2, sigma=0.0, random_state=myrandstate) * binaryimager 
            #, spatial_prior = labeled ) * binaryimager 
            imggk = ants.iMath( binaryimager, "PropagateLabelsThroughMask", imggk, 200000, 0 )
            sum2 = ants.label_overlap_measures(ants.threshold_image(imggk,2,2), labeledTemp) .MeanOverlap[0]
            sum1 = ants.label_overlap_measures(ants.threshold_image(imggk,1,1), labeledTemp) .MeanOverlap[0]
            voldiff = abs( ants.threshold_image(imggk,2,2).sum() - ants.threshold_image(imggk,1,1).sum() )
            if ( sum1 > sum2 and sum1 > bestsum and voldiff < bestvol ) :
                if verbose:
                    print("best " + str(myrandstate) + " diff " + str(voldiff) )
                isbest=True
                bestsum = sum1
                kmeansLabel=1 
                imggkbest = imggk.clone()
                bestvol = voldiff
            elif ( sum2 > bestsum and voldiff < bestvol ): 
                isbest=True
                if verbose:
                    print("best " + str(myrandstate) + " diff " + str(voldiff) )
                bestsum = sum2
                kmeansLabel=2
                imggkbest = imggk.clone()
                bestvol = voldiff
            if plot and isbest :
                ants.plot( binaryimager, ants.threshold_image(imggk,kmeansLabel,kmeansLabel), axis=2, crop=True )
        imggk = imggkbest
        labeled = remove_curvature_spine( curvitr, 
            ants.threshold_image(imggk,kmeansLabel,kmeansLabel) )
    else:
        labeled = remove_curvature_spine( curvitr, labeled )
    if plot :
        ants.plot( binaryimager, labeled, axis=2, crop=True )
#    labeled = ants.iMath( sidelabelRm * ants.threshold_image(imggk,kmeansLabel,kmeansLabel), 
#            "PropagateLabelsThroughMask", 
#            labeled * ants.threshold_image(imggk,kmeansLabel,kmeansLabel), 200000, 0 )
#    if plot:
#        ants.plot( binaryimager, labeled, axis=2, crop=True )
    if ventricle_label is not None:
        ventgrow = ants.threshold_image( segmentation, ventricle_label, ventricle_label ).iMath("MD",1)
        ventgrow = ants.resample_image_to_target( ventgrow, labeled, interp_type='nearestNeighbor' )
        labeled = labeled * ventgrow
    labeled[ curvitr == 0 ] = 0.0
    # now apply the subdivision of the labels:
    if subdivide > 0 :
        for x in range(subdivide):
            labeled = antspyt1w.subdivide_labels(labeled)
    elif grid > 0:
        gridder = tuple( [True] * labeled.dimension )
        gg = ants.create_warped_grid( labeled*0, grid_step=grid, grid_width=1, grid_directions=gridder )
        gg = gg * labeled
        gg = ants.label_clusters( gg, 1 )
        labeled = ants.iMath( labeled, 'PropagateLabelsThroughMask', gg, 1000, 0)
        if verbose:
            print("end prop ") 
    elif priorparcellation is not None:
        print("Use prior parcellation")
        priorsmapped = ants.apply_transforms( curvitr, priorparcellation, reg['fwdtransforms'], 
            interpolator='nearestNeighbor' )
        labeled = ants.iMath( ants.threshold_image(labeled,1,9.e9), 
            'PropagateLabelsThroughMask', priorsmapped, 200000, 0 )
    if verbose:
        print( np.unique( labeled.numpy() ) )
        print( str( labeled.sum()  )  )
    mydf = make_label_dataframe( labeled )
    descriptor = antspyt1w.map_intensity_to_dataframe( mydf, curvitr, labeled )
    descriptor = compute_geom_per_label( labeled, descriptor, flatness, 'Flatness')
    gdesc = ants.label_geometry_measures( labeled )
    geos = ['Mean', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared', 'Eccentricity', 'Elongation','Flatness']
    descriptor = pd.concat([descriptor.reset_index(drop=True), gdesc.reset_index(drop=True)], axis=1)
    descriptor = pd_to_wide( descriptor, column_values=geos)
    descriptor = descriptor.loc[:, ~descriptor.columns.str.startswith('nan')]
    return curvitr, labeled, descriptor

def shape_split_thickness(two_label_segmentation, g=1, w=2, verbose=False):
    """
    Compute thickness-based shape splitting using the Kelly Kapowski algorithm.

    This function takes a two-label segmentation image and computes the thickness map 
    between two regions, typically representing gray matter (GM) and white matter (WM), 
    using the `ants.kelly_kapowski` function.

    Parameters
    ----------
    two_label_segmentation : ants.ANTsImage
        A segmentation image with two distinct labels, typically representing different 
        anatomical regions (e.g., gray matter and white matter).
    
    g : int, optional
        The label representing the "g" region, typically gray matter. Default is 1.
    
    w : int, optional
        The label representing the "w" region, typically white matter. Default is 2.
    
    verbose : bool, optional
        If True, additional information about the process will be printed. Default is False.

    Returns
    -------
    ants.ANTsImage
        An ANTsImage object representing the computed thickness map between the specified 
        regions.

    Notes
    -----
    - The Kelly Kapowski algorithm computes a Laplacian thickness map between regions.
    - The `two_label_segmentation` input must be a segmentation with exactly two distinct labels.

    Example
    -------
    >>> import ants
    >>> import curvanato
    >>> two_label_segmentation = ants.image_read("segmentation.nii.gz")
    >>> thickness_map = shape_split_thickness(two_label_segmentation, g=1, w=2, verbose=True)
    >>> ants.plot(thickness_map)

    """
    gmimg = ants.threshold_image(two_label_segmentation, g, g)
    wmimg = ants.threshold_image(two_label_segmentation, w, w)
    myverb = 0
    if verbose:
        myverb=1
    mykk = ants.kelly_kapowski(
        two_label_segmentation,
        g=gmimg,
        w=wmimg,
        its=45,
        r=0.025,
        m=1.5,
        gm_label=g,
        wm_label=w,
        verbose=myverb
    )
    return mykk

def compute_geom_per_label(img, dataframe, geom_fn, geom_name):
    """
    Compute a flatness metric for each label in a segmentation image and update the DataFrame.

    Parameters:
    - img: ANTsPy image. The input image with labeled segmentation regions.
    - dataframe: pandas.DataFrame. Must contain a column 'Label' with label values.
    - geom_fn: function. A user-defined function that computes flatness for a given binary image.
    - geom_name: name for the new column

    Returns:
    - dataframe: pandas.DataFrame. Updated DataFrame with a new 'Flatness' column.
    """
    if 'Label' not in dataframe.columns:
        raise ValueError("The dataframe must contain a 'Label' column.")

    # Initialize the Flatness column if it doesn't exist
    if geom_name not in dataframe.columns:
        dataframe[geom_name] = None

    # Loop over unique label values in the DataFrame
    for i, label in enumerate(dataframe['Label']):
        # Threshold the image for the current label
        temp = ants.threshold_image(img, label, label)
        
        # Compute the flatness for the thresholded region
        ff = geom_fn(temp)
        
        # Update the DataFrame
        dataframe.at[i, geom_name] = ff

    return dataframe

def compute_smoothing_spacing(segmentation):
    """
    Computes a smoothing factor based on the spacing of the segmentation image.

    Parameters:
    - segmentation: ANTsPy image. The input segmentation image.

    Returns:
    - smoothing: float. The computed smoothing factor.
    """
    spmag = sum(sp ** 2 for sp in ants.get_spacing(segmentation))
    return np.sqrt(spmag)

def load_caudate_labels(prior_labels, prior_target_label):
    """
    Loads caudate region labels with specified configurations.

    Parameters:
    - prior_labels: list of int. Labels to load for caudate regions.
    - prior_target_label: list of int. Labels to subdivide.

    Returns:
    - caud0: ANTsPy image. Loaded labels for the prior caudate.
    - caudsd: ANTsPy image. Subdivided target caudate labels.
    """
    caud0 = load_labeled_caudate(label=prior_labels, subdivide=0, option='laterality')
    caudsd = load_labeled_caudate(label=prior_target_label, subdivide=0, grid=0, option='laterality')
    return caud0, caudsd

def compute_curvature_and_resample(binary_image, labeled, smoothing):
    """
    Computes curvature and resamples the image to match the target.

    Parameters:
    - binary_image: ANTsPy image. Binary image of the target region.
    - labeled: ANTsPy image. Labeled image for reference.
    - smoothing: float. Smoothing factor for curvature computation.

    Returns:
    - curvit: ANTsPy image. Curvature image.
    - curvitr: ANTsPy image. Resampled curvature image.
    - binary_resampled: ANTsPy image. Resampled binary image.
    """
    curvit = compute_curvature(binary_image, smoothing=smoothing, distance_map=True)
    curvitr = ants.resample_image_to_target(curvit, labeled, interp_type='linear')
    binary_resampled = ants.resample_image_to_target(binary_image, labeled, interp_type='nearestNeighbor')
    return curvit, curvitr, binary_resampled

def cluster_image_gradient_prop(binary_image, n_clusters=2, sigma=0.25):
    """
    Clusters the gradient of a binary image.

    Parameters:
    - binary_image: ANTsPy image. Binary image of the target region.
    - n_clusters: int. Number of clusters for k-means.
    - sigma: float. Smoothing factor for gradient computation.

    Returns:
    - clustered_image: ANTsPy image. Image with clustered gradient.
    """
    imggk = cluster_image_gradient(binary_image, binary_image, n_clusters=n_clusters, sigma=sigma)
    return ants.iMath(binary_image, "PropagateLabelsThroughMask", imggk, 200000, 0)

def determine_dominant_cluster(clustered_image, labeled):
    """
    Determines the dominant cluster label based on overlaps with labeled regions.

    Parameters:
    - clustered_image: ANTsPy image. Clustered gradient image.
    - labeled: ANTsPy image. Labeled reference image.

    Returns:
    - dominant_label: int. Label of the dominant cluster.
    """
    sum2 = (ants.threshold_image(clustered_image, 2, 2) * labeled).sum()
    sum1 = (ants.threshold_image(clustered_image, 1, 1) * labeled).sum()
    return 1 if sum1 > sum2 else 2

def remove_spine_and_finalize_labels(curvitr, clustered_image, labeled, dominant_label):
    """
    Removes the spine region and propagates labels through the mask.

    Parameters:
    - curvitr: ANTsPy image. Resampled curvature image.
    - clustered_image: ANTsPy image. Clustered gradient image.
    - labeled: ANTsPy image. Labeled reference image.
    - dominant_label: int. Dominant cluster label.

    Returns:
    - final_labeled: ANTsPy image. Final labeled image.
    """
    sidelabel_rm = remove_curvature_spine(curvitr, ants.threshold_image(clustered_image, dominant_label, dominant_label))
    final_labeled = ants.iMath(
        sidelabel_rm * ants.threshold_image(clustered_image, dominant_label, dominant_label),
        "PropagateLabelsThroughMask",
        labeled * ants.threshold_image(clustered_image, dominant_label, dominant_label),
        200000,
        0
    )
    return final_labeled

