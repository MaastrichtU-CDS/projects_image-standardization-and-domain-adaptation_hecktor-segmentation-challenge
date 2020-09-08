import numpy as np
from scipy import ndimage


def get_cc_array(binary_array, cc_structure=None):
    """
    Returns a label map with a unique integer label for each connected geometrical object in the given binary array.
    Integer labels of components start from 1. Background is 0.
    """

    if not cc_structure:  # If not given, set 26-connected structure as default
        cc_structure = np.array([
                                    [[1,1,1],
                                     [1,1,1],
                                     [1,1,1]],

                                    [[1,1,1],
                                     [1,1,1],
                                     [1,1,1]],

                                    [[1,1,1],
                                     [1,1,1],
                                     [1,1,1]]
                                   ])
        # or alternatively, use the following function -
        # cc_structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    cc_labeled_array, num_cc = ndimage.label(binary_array, structure=cc_structure)

    print("Number of connected components found:", num_cc)
    return cc_labeled_array


def extract_and_enhance_gtv_mask(pred_mask_np, closing_structure=None, remove_brain_fp=False):
    """
    Post-processing function: Connected components + morphological operations

    Params:
        - pred_mask_np: Numpy array of shape (Depth, Height, Width). Binary GTV mask produced as the thresholded output of the neural-network
        - structure_matrix: Numpy array. Structuring element for morphological closing
        - remove_brain_fp: Bool.
    Returns:
        - enhanced_gtv_mask_np: Numpy array of shape (Depth, Height, Width). Post-processed binary GTV mask
    """

    # Apply morphological binary closing to remove small gaps in the mask -----

    if not closing_structure: # If not given, define a 5x5 structure matrix with a some-what spherical shape
        closing_structure = np.array([
                                     [[0,1,1,1,0],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [0,1,1,1,0]],

                                     [[0,1,1,1,0],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [0,1,1,1,0]],

                                     [[0,1,1,1,0],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [1,1,1,1,1],
                                      [0,1,1,1,0]]
                                    ])

    pred_mask_closed_np = ndimage.binary_closing(pred_mask_np, structure=closing_structure)


    # Get label map of connected components -----
    pred_mask_cc_np = get_cc_array(pred_mask_closed_np)


    # If enabled, remove the false positive in the brain region -----
    if remove_brain_fp:
    	# Brain-region false positives are assumed to exist beyond axial slice 41, based on the Slice v/s #positive voxels bar plots
        pred_mask_cc_np[41:,:,:] = 0


    # Obtain the GTV as the largest geometric object -----
    cc_voxel_count = [np.sum(pred_mask_cc_np==label) for label in range(pred_mask_cc_np.max())]
    gtv_label = np.argmax(cc_voxel_count[1:]) + 1
    postproc_gtv_mask_np = pred_mask_cc_np == gtv_label

    return postproc_gtv_mask_np